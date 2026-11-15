"""Strict structured-MLIR subset parser/interpreter, independent of TIR frontend.

This is not the official MLIR verifier. Unknown syntax, side-effectful symbolic
branches, calls, layouts and unsupported operations stop execution explicitly.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field

import numpy as np

from .execution import Buffer, Expr, Unsupported, apply, cast, dtype, select, storage_type, trace_value, literal, merge_trace, mlir_cast, mlir_cmpi
from .capabilities import MLIR_OPS

SSA = r"%[\w.$-]+"
MEMREF = r"memref<(?:(?:\d+|\?)x)*\w+(?:, strided<\[(?:[?\d]+(?:, )?)*\](?:, offset: [?\d]+)?>)?>"

def split_types(text):
    parts=[];depth=0;start=0
    for i,c in enumerate(text):
        if c in '<[(':depth+=1
        elif c in '>])':depth-=1
        elif c==',' and depth==0:parts.append(text[start:i].strip());start=i+1
    parts.append(text[start:].strip())
    return parts

def memref_type(text):
    m=re.fullmatch(r"memref<((?:(?:\d+|\?)x)*)(\w+)(?:, strided<\[([^]]+)\](?:, offset: ([?\d]+))?>)?>",text)
    if not m:raise Unsupported('memref 布局格式尚未支持')
    dims=m[1].rstrip('x').split('x') if m[1] else []
    return dims,m[2],m[3],m[4]



@dataclass
class Node:
    text: str
    line: int
    body: list = field(default_factory=list)
    otherwise: list = field(default_factory=list)


def _parse(source):
    root = []
    stack = [(None, root)]
    for number, raw in enumerate(source.splitlines(), 1):
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        if line == "} else {":
            if len(stack) < 2 or "scf.if" not in stack[-1][0].text:
                raise Unsupported(f"L{number}: 无法解析 else")
            node = stack.pop()[0]
            stack.append((node, node.otherwise))
            continue
        if line == "}":
            if len(stack) == 1:
                raise Unsupported("MLIR region 括号不平衡")
            stack.pop()
            continue
        node = Node(line, number)
        stack[-1][1].append(node)
        if line.endswith("{"):
            stack.append((node, node.body))
        elif "{" in line or "}" in line:
            raise Unsupported(f"L{number}: 不支持的内联 region/属性格式")
    if len(stack) != 1:
        raise Unsupported("MLIR region 未闭合")
    if len(root) != 1 or root[0].text != "module {":
        raise Unsupported("MLIR 执行需要单个无属性 module")
    functions = root[0].body
    if len(functions) != 1 or not functions[0].text.startswith("func.func @"):
        raise Unsupported("MLIR 执行需要单个公开函数且不含外部调用")
    return functions[0]


class MLIRInterpreter:
    OPS = MLIR_OPS

    def __init__(self, source):
        self.function = _parse(source)
        match = re.fullmatch(r"func\.func @([\w.$-]+)\((.*)\) \{", self.function.text)
        if match is None:
            raise Unsupported("未支持的 MLIR 函数头格式")
        self.symbol = match[1]
        self.parameters = []
        for part in split_types(match[2]) if match[2] else []:
            arg = re.fullmatch(rf"\s*({SSA}): ({MEMREF})\s*", part)
            if arg is None:raise Unsupported("仅支持 memref 参数")
            shape,kind,strides,offset=memref_type(arg[2])
            self.parameters.append((arg[1], shape, kind))
            # Validate known static stride contracts on run(), do not erase layouts.
            if not hasattr(self,'layouts'):self.layouts={}
            self.layouts[arg[1]]=(strides,offset)
        self._preflight(self.function.body)

    def _preflight(self, nodes):
        for node in nodes:
            text = re.sub(rf"^{SSA}\s*=\s*", "", node.text)
            token = re.match(r"[\w.]+",text)
            op = token[0] if token else ""
            if op not in self.OPS:
                raise Unsupported(f"L{node.line}: 未支持 MLIR 操作 {op}")
            patterns={
                'scf.for':rf'scf.for {SSA} = {SSA} to {SSA} step {SSA} \{{',
                'scf.parallel':r'scf.parallel \([^)]+\) = \([^)]+\) to \([^)]+\) step \([^)]+\) \{',
                'scf.if':rf'scf.if {SSA}(?: -> \(\w+\))? \{{',
                'scf.yield':rf'scf.yield(?: {SSA} : \w+)?',
                'scf.reduce':r'scf.reduce', 'return':r'return',
                'memref.alloca':rf'memref.alloca\([^)]*\) : {MEMREF}',
                'memref.alloc':rf'memref.alloc\([^)]*\) : {MEMREF}',
                'memref.load':rf'memref.load {SSA}\[[^]]*\] : {MEMREF}',
                'memref.store':rf'memref.store {SSA}, {SSA}\[[^]]*\] : {MEMREF}',
                'memref.dim':rf'memref.dim {SSA}, {SSA} : {MEMREF}',
                'arith.constant':r'arith.constant (?:true|false|[^ ]+ : \w+)',
                'arith.cmpi':rf'arith.cmpi (?:eq|ne|[su](?:lt|le|gt|ge)), {SSA}, {SSA} : \w+',
                'arith.select':rf'arith.select {SSA}, {SSA}, {SSA} : \w+',
                'linalg.fill':rf'linalg.fill ins\({SSA} : \w+\) outs\({SSA} : {MEMREF}\)',
            }
            pattern=patterns.get(op)
            if op=='arith.negf':pattern=rf'arith.negf {SSA} : \w+'
            if op=='memref.copy':pattern=rf'memref.copy {SSA}, {SSA} : {MEMREF} to {MEMREF}'
            if op in {'math.exp','math.exp2','math.rsqrt','math.sqrt'}:pattern=rf'math\.\w+ {SSA} : \w+'
            if op=='arith.cmpf':pattern=rf'arith.cmpf \w+, {SSA}, {SSA} : \w+'
            if op=='memref.subview':pattern=rf'memref.subview {SSA}\[[^]]*\] \[[^]]*\] \[[^]]*\] : {MEMREF} to {MEMREF}'
            if op in {'arith.index_cast','arith.index_castui','arith.extf','arith.truncf',
                       'arith.sitofp','arith.uitofp','arith.extsi','arith.extui','arith.trunci'}:
                pattern=rf'arith\.\w+ {SSA} : \w+ to \w+'
            if op in {'arith.addi','arith.subi','arith.muli','arith.addf','arith.subf','arith.mulf','arith.divf','arith.andi','arith.ori','arith.floordivsi','arith.remsi'}:
                pattern=rf'arith\.\w+ {SSA}, {SSA} : \w+'
            if not pattern or not re.fullmatch(pattern,text):
                raise Unsupported(f'L{node.line}: 未支持 {op} 的具体语法')
            if op in {'arith.andi','arith.ori'} and not text.endswith(' : i1'):
                raise Unsupported('当前仅支持 i1 的 andi/ori')
            for type_name in re.findall(r'\b(?:bf16|f16|f32|f64|i1|i8|i16|i32|i64|index)\b',text):
                dtype(type_name)
            if node.body and op not in {"scf.for", "scf.parallel", "scf.if"}:
                raise Unsupported(f"L{node.line}: 未支持的 region")
            self._preflight(node.body)
            self._preflight(node.otherwise)

    def run(self, buffers, symbols=None, symbolic=False):
        if len(buffers) != len(self.parameters):
            raise Unsupported("MLIR 参数数量与测试契约不符")
        self.symbolic = symbolic
        self.budget = buffers[0].budget
        self.pure_only = False
        env = {}
        for (param, shape, kind), buf in zip(self.parameters, buffers):
            if len(shape) != len(buf.shape) or any(d != "?" and int(d) != n for d,n in zip(shape,buf.shape)):
                raise Unsupported("MLIR 参数 shape 与测试契约不符")
            if storage_type(dtype(kind)) != storage_type(buf.dtype):
                raise Unsupported("MLIR 参数 dtype 与测试契约不符")
            strides,offset=self.layouts[param]
            if strides is not None:
                strides=[v.strip() for v in strides.split(',')]
                actual=[s//buf.dtype.itemsize for s in buf.values.strides]
                if len(strides)!=len(actual) or any(s!='?' and int(s)!=v for s,v in zip(strides,actual)):
                    raise Unsupported('MLIR 静态 stride 与参数布局不符')
            if offset not in (None,'?','0'):raise Unsupported('非零参数 offset 需要显式 view 契约')
            env[param] = buf
            buf.name = param
        self.sequence(self.function.body, env)
        return {"steps": self.budget.steps, "execution_model": "structured logical loops, serial enumeration of parallel iterations"}

    def sequence(self, nodes, env):
        for node in nodes:
            self.budget.tick()
            with self.budget.at('operation', node.line):
                result = self.operation(node, env)
            if isinstance(result, tuple) and result[0] == "yield":
                return result[1]
        return None

    def value(self, token, env):
        token = token.strip()
        if token not in env:
            raise Unsupported(f"SSA 值未定义：{token}")
        return env[token]

    def indices(self, text, env):
        return tuple(self.value(v, env) for v in text.split(",")) if text.strip() else ()

    def operation(self, node, env):
        original, line = node.text, node.line
        assignment = re.match(rf"^({SSA})\s*=\s*(.*)", original)
        target = assignment[1] if assignment else None
        text = assignment[2] if assignment else original
        if text.startswith("scf.for "):
            m = re.fullmatch(rf"scf.for ({SSA}) = ({SSA}) to ({SSA}) step ({SSA}) \{{", text)
            if not m:
                raise Unsupported(f"L{line}: 未支持的 scf.for 格式")
            lower, upper, step = [int(self.value(m[i],env)) for i in (2,3,4)]
            if step <= 0:
                raise Unsupported("scf.for 需要正步长")
            for value in range(lower, upper, step):
                local = dict(env)
                local[m[1]] = np.int64(value)
                with self.budget.at('loop', line, variable=m[1], value=value, lower=lower, upper=upper, step=step):
                    self.sequence(node.body, local)
            return None
        if text.startswith("scf.parallel "):
            m = re.fullmatch(r"scf.parallel \(([^)]+)\) = \(([^)]+)\) to \(([^)]+)\) step \(([^)]+)\) \{", text)
            if not m:
                raise Unsupported(f"L{line}: 未支持的 scf.parallel")
            names = [v.strip() for v in m[1].split(",")]
            bounds = [self.indices(m[i],env) for i in (2,3,4)]
            if any(len(b) != len(names) for b in bounds) or any(int(s) <= 0 for s in bounds[2]):
                raise Unsupported("parallel 循环维度或步长错误")
            region=self.budget.new_region()
            for index in itertools.product(*[range(int(a),int(b),int(s)) for a,b,s in zip(*bounds)]):
                local = dict(env)
                local.update({k:np.int64(v) for k,v in zip(names,index)})
                with self.budget.iteration(region,index):
                    with self.budget.at('parallel', line, variables=names, values=list(index)):
                        self.sequence(node.body,local)
            return None
        if text.startswith("scf.if "):
            m = re.fullmatch(rf"scf.if ({SSA})(?: -> \(([\w]+)\))? \{{", text)
            if not m:
                raise Unsupported(f"L{line}: 未支持的 scf.if")
            cond = self.value(m[1],env)
            if isinstance(cond, Expr):
                if not target or not node.otherwise:
                    raise Unsupported("输入相关符号 if 必须为无副作用标量选择")
                old = self.pure_only
                self.pure_only = True
                try:
                    yes = self.sequence(node.body,dict(env))
                    no = self.sequence(node.otherwise,dict(env))
                finally:
                    self.pure_only = old
                value = select(cond,yes,no)
            else:
                with self.budget.at('guard', line, condition=m[1], value=bool(cond)):
                    value = self.sequence(node.body if bool(cond) else node.otherwise,dict(env))
            if target:
                if value is None:
                    raise Unsupported("if 缺少标量 yield")
                env[target] = value
            return None
        if text in {"scf.yield", "scf.reduce", "return"}:
            return None
        if text.startswith("scf.yield "):
            m = re.fullmatch(rf"scf.yield ({SSA}) : [\w]+",text)
            if not m:
                raise Unsupported("只支持单值 yield")
            return ("yield",self.value(m[1],env))
        if text.startswith("memref.store "):
            if self.pure_only:
                raise Unsupported("符号 if 分支中含 store")
            m = re.fullmatch(rf"memref.store ({SSA}), ({SSA})\[([^]]*)\] : {MEMREF}",text)
            if not m:
                raise Unsupported(f"L{line}: store 格式未支持")
            self.value(m[2],env).write(self.indices(m[3],env),self.value(m[1],env),line)
            return None
        if text.startswith('memref.copy '):
            if self.pure_only:raise Unsupported('符号分支含 memref.copy')
            m=re.fullmatch(rf'memref.copy ({SSA}), ({SSA}) : {MEMREF} to {MEMREF}',text)
            source,target=[self.value(m[i],env) for i in (1,2)]
            if source.shape!=target.shape or source.dtype!=target.dtype:raise Unsupported('memref.copy 的 shape/dtype 不同')
            if np.shares_memory(source.values,target.values):raise Unsupported('memref.copy 重叠存储尚未支持')
            for idx in np.ndindex(source.shape):target.write(idx,source.read(idx,line),line)
            return None
        if text.startswith("linalg.fill "):
            if self.pure_only:
                raise Unsupported("符号分支含 fill")
            m = re.fullmatch(rf"linalg.fill ins\(({SSA}) : [\w]+\) outs\(({SSA}) : memref<[^>]+>\)",text)
            if not m:
                raise Unsupported("仅支持标量 linalg.fill")
            buf = self.value(m[2],env)
            for index in np.ndindex(buf.shape):
                buf.write(index,self.value(m[1],env),line)
            return None
        if target is None:
            raise Unsupported(f"L{line}: 未支持的无返回值操作 {text}")
        if text.startswith(("memref.alloca", "memref.alloc(")):
            if self.pure_only:
                raise Unsupported("符号分支含 allocation")
            m = re.fullmatch(r"memref.alloca?\(([^)]*)\) : memref<((?:(?:\d+|\?)x)*)([\w]+)>",text)
            if not m:
                raise Unsupported("allocation 格式未支持")
            parts = m[2].rstrip("x").split("x") if m[2] else []
            supplied=self.indices(m[1],env)
            if len(supplied)!=parts.count('?'):raise Unsupported('allocation 动态尺寸数量不匹配')
            dynamic = iter(supplied)
            shape = tuple(int(next(dynamic)) if v == "?" else int(v) for v in parts)
            self.budget.shape(shape)
            value = Buffer(np.empty(shape,dtype=dtype(m[3])),initialized=False,
                           symbolic=self.symbolic,budget=self.budget,name=target,allocation_line=line)
        elif text.startswith("memref.load "):
            m = re.fullmatch(rf"memref.load ({SSA})\[([^]]*)\] : ({MEMREF})",text)
            if not m:
                raise Unsupported("load 格式未支持")
            value = cast(self.value(m[1],env).read(self.indices(m[2],env),line),memref_type(m[3])[1])
        elif text.startswith("memref.dim "):
            m = re.fullmatch(rf"memref.dim ({SSA}), ({SSA}) : {MEMREF}",text)
            if not m:
                raise Unsupported("dim 格式未支持")
            value = np.int64(self.value(m[1],env).shape[int(self.value(m[2],env))])
        elif text.startswith('memref.subview '):
            m=re.fullmatch(rf'memref.subview ({SSA})\[([^]]*)\] \[([^]]*)\] \[([^]]*)\] : ({MEMREF}) to ({MEMREF})',text)
            if not m:raise Unsupported(f'L{line}: subview 语法未支持')
            base=self.value(m[1],env)
            def nums(t):return tuple(int(self.value(v.strip(),env)) if v.strip().startswith('%') else int(v) for v in t.split(','))
            offsets,sizes,strides=[nums(m[i]) for i in (2,3,4)]
            dims,kind,layout,offset=memref_type(m[6])
            # Rank-reducing subview must have one unambiguous size-1 projection.
            if len(dims)>len(sizes):raise Unsupported('subview 目标 rank 大于输入 rank')
            choices=[]
            for drop in itertools.combinations([i for i,n in enumerate(sizes) if n==1],len(sizes)-len(dims)):
                kept=[n for i,n in enumerate(sizes) if i not in drop]
                if all(d=='?' or int(d)==n for d,n in zip(dims,kept)):choices.append(drop)
            if len(choices)!=1:raise Unsupported('subview 降维映射不唯一')
            value=base.subview(offsets,sizes,strides,choices[0])
            if dtype(kind)!=value.dtype:raise Unsupported('subview 不能改变 dtype')
            if offset not in (None,'?'):
                root=base
                while hasattr(root,'_parent'):root=root._parent
                actual_offset=(value.values.ctypes.data-root.values.ctypes.data)//value.dtype.itemsize
                if int(offset)!=actual_offset:raise Unsupported('subview 静态 offset 与实际共享存储不符')
            if layout:
                actual=[v//value.dtype.itemsize for v in value.values.strides]
                if any(s.strip()!='?' and int(s)!=v for s,v in zip(layout.split(','),actual)):
                    raise Unsupported('subview 声明 stride 不匹配')
        elif text.startswith('arith.negf '):
            m=re.fullmatch(rf'arith.negf ({SSA}) : (\w+)',text)
            value=apply('neg',[self.value(m[1],env)],m[2])
        elif text.startswith('math.'):
            m=re.fullmatch(rf'math\.(exp|exp2|rsqrt|sqrt) ({SSA}) : (\w+)',text)
            if not m:raise Unsupported('math 操作属性尚未支持')
            value=apply(m[1],[self.value(m[2],env)],m[3])
        elif text.startswith('arith.cmpf '):
            m=re.fullmatch(rf'arith.cmpf (\w+), ({SSA}), ({SSA}) : (\w+)',text)
            if not m:raise Unsupported('cmpf 语法未支持')
            a,b=self.value(m[2],env),self.value(m[3],env)
            predicates={'oeq':'eq','one':'ne','olt':'lt','ole':'le','ogt':'gt','oge':'ge','ueq':'eq','une':'ne','ult':'lt','ule':'le','ugt':'gt','uge':'ge'}
            pred=m[1]
            if pred not in predicates and pred not in {'ord','uno','true','false'}:raise Unsupported('未知 cmpf 谓词')
            if isinstance(a,Expr) or isinstance(b,Expr):
                nodes=(literal(a),literal(b));value=Expr('cmpf_'+pred,nodes,'i1',merge_trace(nodes))
            else:
                unordered=bool(np.isnan(a) or np.isnan(b))
                if pred in {'true','false','ord','uno'}:value=np.bool_({'true':True,'false':False,'ord':not unordered,'uno':unordered}[pred])
                else:value=np.bool_((unordered or bool(apply(predicates[pred],[a,b]))) if pred.startswith('u') else (not unordered and bool(apply(predicates[pred],[a,b]))))
        elif text.startswith("arith.constant "):
            m = re.fullmatch(r"arith.constant (.+?)(?: : (\w+))?",text)
            token,kind = m[1],m[2]
            if token in {"true","false"}:
                value = np.bool_(token == "true")
            elif kind:
                if token.startswith("0x") and (dtype(kind).kind == "f" or dtype(kind).name in {"bfloat16", "float8_e4m3fn"}):
                    value = np.array(int(token,16),dtype="uint"+str(dtype(kind).itemsize*8)).view(dtype(kind))[()]
                else:
                    try:
                        value = np.array(float(token) if dtype(kind).name in {"bfloat16", "float8_e4m3fn"} else token,dtype=dtype(kind))[()]
                    except (TypeError,ValueError,OverflowError) as exc:
                        raise Unsupported(f'L{line}: constant 位模式或文本格式未支持：{token}') from exc
            else:
                raise Unsupported("constant 缺少支持的类型")
        elif text.startswith(tuple('arith.'+op+' ' for op in ('index_cast','index_castui','extf','truncf','sitofp','uitofp','extsi','extui','trunci'))):
            m = re.fullmatch(rf"arith\.(\w+) ({SSA}) : (\w+) to (\w+)",text)
            if not m:
                raise Unsupported("cast 格式未支持")
            value = mlir_cast(m[1],self.value(m[2],env),m[3],m[4])
        elif text.startswith("arith.cmpi "):
            m = re.fullmatch(rf"arith.cmpi (\w+), ({SSA}), ({SSA}) : (\w+)",text)
            if not m:
                raise Unsupported("cmpi 格式未支持")
            value = mlir_cmpi(m[1],self.value(m[2],env),self.value(m[3],env),m[4])
        elif text.startswith("arith.select "):
            m = re.fullmatch(rf"arith.select ({SSA}), ({SSA}), ({SSA}) : (\w+)",text)
            if not m:
                raise Unsupported("select 格式未支持")
            value = select(*[self.value(m[i],env) for i in (1,2,3)])
        else:
            m = re.fullmatch(rf"arith\.(addi|subi|muli|addf|subf|mulf|divf|andi|ori|floordivsi|remsi) ({SSA}), ({SSA}) : (\w+)",text)
            if not m:
                raise Unsupported(f"L{line}: 未支持格式 {text}")
            op = {"addi":"add","subi":"sub","muli":"mul","addf":"add","subf":"sub","mulf":"mul","divf":"div","andi":"and","ori":"or","floordivsi":"floordiv","remsi":"rem"}[m[1]]
            value = apply(op,[self.value(m[2],env),self.value(m[3],env)],m[4])
        env[target] = trace_value(value, target, line, text.split()[0])
        return None

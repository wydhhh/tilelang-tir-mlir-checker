"""Bounded TVMScript AST interpreter; never eval/exec/imports input source."""
from __future__ import annotations

import ast
import itertools
from contextlib import nullcontext

import numpy as np

from .execution import Buffer, Budget, Expr, SemanticFault, Unsupported, apply, cast, dtype, select, trace_value
from .capabilities import TIR_METADATA as METADATA, TIR_CALLS as CALLS, TIR_SCALARS as SCALARS


def name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return name(node.value) + "." + node.attr
    return ""


class TIRInterpreter:
    def __init__(self, source, *, max_steps=2_000_000, metadata=None):
        self.metadata=metadata
        try:
            self.tree = ast.parse(source)
        except SyntaxError as exc:
            raise Unsupported(f"TIR Python 语法不可解析：{exc}") from exc
        funcs = [n for n in ast.walk(self.tree) if isinstance(n, ast.FunctionDef)
                 and any(name(d) == "T.prim_func" for d in n.decorator_list)]
        if len(funcs) != 1:
            raise Unsupported("执行后端只接受一个 @T.prim_func")
        self.function = funcs[0]
        self.max_steps = max_steps
        self._preflight(self.function.body)
        for node in self.tree.body:
            if not isinstance(node, (ast.ClassDef, ast.ImportFrom)):
                raise Unsupported(f"L{node.lineno}: 不支持的 TIR 顶层语句")

    def _preflight(self, statements):
        for stmt in statements:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) and name(stmt.value.func) in METADATA:
                continue
            if isinstance(stmt, ast.With):
                if len(stmt.items) != 1 or name(stmt.items[0].context_expr.func) != "T.block":
                    raise Unsupported(f"L{stmt.lineno}: 不支持的 with")
                self._preflight(stmt.body)
            elif isinstance(stmt, (ast.For, ast.If)):
                self._preflight(stmt.body)
                self._preflight(stmt.orelse)
                expr = stmt.iter if isinstance(stmt, ast.For) else stmt.test
                self._preflight_expr(expr)
            elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr)):
                for node in ast.walk(stmt):
                    if isinstance(node, ast.Call) and name(node.func) not in CALLS | SCALARS:
                        raise Unsupported(f"L{node.lineno}: 未支持调用 {name(node.func)}")
                self._preflight_expr(stmt)
            elif isinstance(stmt, ast.Pass):
                pass
            else:
                raise Unsupported(f"L{stmt.lineno}: 未支持语句 {type(stmt).__name__}")

    def _preflight_expr(self, expr):
        for node in ast.walk(expr):
            if isinstance(node, ast.Call) and name(node.func) not in CALLS | SCALARS:
                raise Unsupported(f"L{node.lineno}: 未支持调用 {name(node.func)}")
            if isinstance(node,ast.BinOp) and type(node.op) not in {ast.Add,ast.Sub,ast.Mult,ast.Div,ast.FloorDiv,ast.Mod}:
                raise Unsupported(f"L{node.lineno}: 未支持二元运算 {type(node.op).__name__}")
            if (isinstance(node,ast.UnaryOp) and not isinstance(node.op,(ast.USub,ast.Not))):
                raise Unsupported(f"L{node.lineno}: 未支持的布尔/一元表达式")
            if isinstance(node,ast.Compare) and (len(node.ops)!=1 or type(node.ops[0]) not in {ast.Eq,ast.NotEq,ast.Lt,ast.LtE,ast.Gt,ast.GtE}):
                raise Unsupported(f"L{node.lineno}: 未支持的比较")
            if isinstance(node,ast.Call) and name(node.func)=='T.match_buffer':
                if any(k.arg not in {'dtype','strides'} for k in node.keywords):
                    raise Unsupported(f"L{node.lineno}: 未支持 match_buffer 附加属性")

    def run(self, buffers, symbols=None, symbolic=False):
        self.budget = buffers[0].budget if buffers else Budget(self.max_steps)
        self.symbolic = symbolic
        self.env = dict(symbols or {})
        self.symbols = dict(symbols or {})
        self.reductions = []
        self.handles = {}
        self.shared = {}
        self.thread_active = False
        self.used_physical_threads = False
        params = self.function.args.posonlyargs + self.function.args.args
        if len(params) != len(buffers):
            raise Unsupported("TIR 参数数量与测试契约不同")
        self.env.update({p.arg: buf for p, buf in zip(params, buffers)})
        self._bind_dynamic_shapes(buffers)
        self._sequence(self.function.body)
        model = ("logical blocks; one-dimensional cooperative thread simulation"
                 if self.used_physical_threads else "logical_blocks; unused physical thread axes only")
        return {"steps": self.budget.steps, "execution_model": model,
                "reductions": self.reductions}

    def _bind_dynamic_shapes(self, buffers):
        # Only direct dimension symbols can be inferred; expressions require
        # explicit symbols in the case contract, never algebraic guesses.
        for stmt in self.function.body:
            if not isinstance(stmt, ast.Assign) or not isinstance(stmt.value, ast.Call):
                continue
            call = stmt.value
            if name(call.func) != "T.match_buffer" or len(call.args) < 2:
                continue
            buf = self.env.get(name(call.args[0]))
            if not isinstance(buf, Buffer):
                continue
            strides=next((k.value for k in call.keywords if k.arg=='strides'),None)
            if isinstance(strides,(ast.Tuple,ast.List)):
                for expression,size in zip(strides.elts,buf.values.strides):
                    symbol=(expression.id if isinstance(expression,ast.Name) else
                            expression.value if isinstance(expression,ast.Constant) and isinstance(expression.value,str) else None)
                    if symbol:
                        size=int(size//buf.dtype.itemsize)
                        old=self.symbols.setdefault(symbol,size)
                        if old!=size:raise Unsupported('stride 符号契约不一致')
            shape = call.args[1]
            if not isinstance(shape, (ast.Tuple, ast.List)) or len(shape.elts) != len(buf.shape):
                raise Unsupported("match_buffer rank 与参数不同")
            for expression, size in zip(shape.elts, buf.shape):
                if isinstance(expression, ast.Name):
                    old = self.symbols.setdefault(expression.id, size)
                    if old != size:
                        raise SemanticFault(f"动态维度 {expression.id} 不一致")
        self.env.update(self.symbols)

    def _sequence(self, statements):
        for offset, stmt in enumerate(statements):
            self.budget.tick()
            if (isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call)
                    and name(stmt.value.func) == "T.launch_thread"):
                call = stmt.value
                axis, extent = self.eval(call.args[0]), int(self.eval(call.args[1]))
                target = name(stmt.targets[0])
                if not target or extent < 0:
                    raise Unsupported("无效的 launch_thread")
                tail = statements[offset + 1:]
                if axis.startswith("blockIdx."):
                    for value in range(extent):
                        self.env[target] = np.int64(value)
                        with self.budget.at('block', stmt.lineno, variable=target, value=value, extent=extent):
                            self._sequence(tail)
                    return
                if not axis.startswith("threadIdx."):
                    raise Unsupported(f"未知线程轴 {axis}")
                used = self._runtime_names(tail)
                if target in used and extent != 1:
                    if self.thread_active:
                        raise Unsupported("当前只支持一维物理线程组")
                    self.used_physical_threads = True
                    self.physical(tail,target,extent)
                    return
                self.env[target] = np.int64(0)
                continue
            self.statement(stmt)

    def _runtime_names(self, statements):
        names = set()
        for stmt in statements:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) and name(stmt.value.func) in METADATA:
                continue
            if isinstance(stmt, ast.With):
                names.update(self._runtime_names(stmt.body))
            else:
                names.update(n.id for n in ast.walk(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load))
        return names

    def statement(self, node):
        with self.budget.at('statement', node.lineno):
            return self._statement(node)

    def _statement(self, node):
        if isinstance(node, ast.With):
            self._sequence(node.body)
        elif isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                raise Unsupported("不支持多目标赋值")
            value = self.eval(node.value)
            self.assign(node.targets[0], value, node.lineno)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            value=self.eval(node.value)
            kind=name(node.annotation)
            if kind.startswith("T."):
                value=cast(value,kind[2:])
            else:
                raise Unsupported("未支持的标量类型注解")
            self.assign(node.target,value,node.lineno)
        elif isinstance(node, ast.AugAssign):
            op = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul"}.get(type(node.op))
            if op is None:
                raise Unsupported("不支持的增量赋值")
            self.assign(node.target, apply(op, [self.eval(node.target), self.eval(node.value)]), node.lineno)
        elif isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Call) and name(node.value.func) in METADATA:
                if name(node.value.func) == "T.block_attr":
                    payload=node.value.args[0]
                    if not isinstance(payload,ast.Dict):raise Unsupported('block_attr 必须为字典')
                    for key,value in zip(payload.keys,payload.values):
                        if isinstance(key,ast.Constant) and key.value=='tl.local_var_init':
                            if not isinstance(value,ast.Dict):raise Unsupported('local_var_init 必须为字典')
                            for handle,initial in zip(value.keys,value.values):self.handles[self.eval(handle)]=self.eval(initial)
                        else:raise Unsupported(f'L{node.lineno}: 未支持 block_attr 属性')
                if name(node.value.func) == "T.attr" and len(node.value.args) > 1 and isinstance(node.value.args[1], ast.Constant) and node.value.args[1].value == "tl.assume":
                    if not bool(self.eval(node.value.args[0])):
                        raise Unsupported(f"L{node.lineno}: 测试不满足 TIR assumption")
                return
            self.eval(node.value)
        elif isinstance(node, ast.For):
            if node.orelse:
                raise Unsupported("不支持循环 else")
            call = node.iter
            if not isinstance(call, ast.Call) or name(call.func) not in {"range", "T.serial", "T.parallel", "T.unroll", "T.vectorized"}:
                raise Unsupported("不支持的循环迭代器")
            bounds = [int(self.eval(v)) for v in call.args]
            region=self.budget.new_region() if name(call.func)=='T.parallel' else None
            for value in range(*bounds):
                self.assign(node.target, np.int64(value), node.lineno)
                with self.budget.iteration(region,value) if region is not None else nullcontext():
                    with self.budget.at('loop', node.lineno, variable=name(node.target), value=value,
                                        bounds=bounds):
                        self._sequence(node.body)
        elif isinstance(node, ast.If):
            cond = self.eval(node.test)
            if isinstance(cond, Expr):
                raise Unsupported("符号执行暂不支持有副作用的输入相关 if；支持标量 if_then_else")
            with self.budget.at('guard', node.lineno, condition=ast.unparse(node.test), value=bool(cond)):
                self._sequence(node.body if bool(cond) else node.orelse)
        elif isinstance(node, ast.Pass):
            return
        else:
            raise Unsupported(f"L{node.lineno}: 不支持语句")

    def assign(self, target, value, line):
        if isinstance(target, ast.Name):
            if value is None:
                if target.id not in self.symbols:
                    raise Unsupported(f"动态符号 {target.id} 未绑定")
                value = self.symbols[target.id]
            self.env[target.id] = value
            if isinstance(value, Buffer):
                value.name = target.id
        elif isinstance(target, ast.Tuple) and isinstance(value, tuple):
            for a, b in zip(target.elts, value):
                self.assign(a, b, line)
        elif isinstance(target, ast.Subscript):
            buf, idx = self.address(target)
            buf.write(idx, value, line)
        else:
            raise Unsupported("不支持的赋值目标")

    def address(self, node):
        buf = self.eval(node.value)
        if not isinstance(buf, Buffer):
            raise Unsupported("非 buffer 下标")
        elements = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if any(isinstance(n, ast.Slice) for n in elements):
            raise Unsupported("当前 TIR 后端仅支持标量下标与 T.region 矩形拷贝")
        return buf, tuple(self.eval(i) for i in elements)

    def eval(self, node):
        value = self._eval(node)
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call)):
            return trace_value(value, ast.unparse(node)[:160], node.lineno,
                               name(node.func) if isinstance(node, ast.Call) else type(node).__name__)
        return value

    def _eval(self, node):
        self.budget.tick()
        if isinstance(node, ast.Constant):
            if isinstance(node.value, float):
                return np.float32(node.value)
            if type(node.value) is int:
                return np.int64(node.value)
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in self.env:
                raise Unsupported(f"L{node.lineno}: 未绑定名称 {node.id}")
            return self.env[node.id]
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(self.eval(n) for n in node.elts)
        if isinstance(node, ast.Subscript):
            buf, idx = self.address(node)
            return buf.read(idx, node.lineno)
        if isinstance(node, ast.BinOp):
            op = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div", ast.FloorDiv: "floordiv", ast.Mod: "mod"}.get(type(node.op))
            if op is None:
                raise Unsupported("不支持的二元运算")
            return apply(op, [self.eval(node.left), self.eval(node.right)])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return apply("neg", [self.eval(node.operand)])
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            op = {ast.NotEq: "ne", ast.Eq: "eq", ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge"}.get(type(node.ops[0]))
            if op is None:
                raise Unsupported("不支持的比较")
            return apply(op, [self.eval(node.left), self.eval(node.comparators[0])])
        if isinstance(node, ast.BoolOp):
            values=[]
            for item in node.values:
                value=self.eval(item)
                values.append(value)
                if not isinstance(value,Expr):
                    if isinstance(node.op,ast.And) and not bool(value): return np.bool_(False)
                    if isinstance(node.op,ast.Or) and bool(value): return np.bool_(True)
            from functools import reduce
            return reduce(lambda a,b:apply('and' if isinstance(node.op,ast.And) else 'or',[a,b]),values)
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.Not):
            return apply('not',[self.eval(node.operand)])
        if isinstance(node, ast.Call):
            return self.call(node)
        raise Unsupported(f"L{getattr(node,'lineno','?')}: 不支持表达式 {type(node).__name__}")

    def call(self, node):
        function = name(node.func)
        if function in {'T.min','T.max'}:
            values=[self.eval(a) for a in node.args]
            if len(values)!=2:raise Unsupported('min/max 需要两个标量')
            if not self.symbolic and any((np.asarray(v).dtype.kind=='f' or np.asarray(v).dtype.name=='bfloat16') and (np.isnan(v) or (v==0 and np.signbit(v))) for v in values):
                raise Unsupported('min/max NaN/负零规则尚未核实')
            return apply(function[2:],values)
        if function == 'T.infinity':
            if len(node.args)!=1 or node.keywords:
                raise Unsupported('T.infinity 需要一个显式浮点类型')
            kind=self.eval(node.args[0])
            if dtype(kind).kind!='f' and dtype(kind).name!='bfloat16':
                raise Unsupported('T.infinity 需要支持无穷大的浮点类型')
            return cast(np.inf,kind)
        if function in {'T.exp','T.exp2','T.rsqrt','T.sqrt','T.sigmoid'}:
            value=self.eval(node.args[0])
            kind=value.type if isinstance(value,Expr) else str(np.asarray(value).dtype)
            return apply(function[2:],[value],kind)
        if function == 'T.handle':
            return ('handle',node.lineno)
        if function == 'T.tvm_storage_sync':
            raise Unsupported(f'L{node.lineno}: barrier 必须在物理线程调度器中执行')
        if function == 'T.reduce':
            return self.reduce(node)
        if function in SCALARS:
            return cast(self.eval(node.args[0]), function[2:]) if node.args else None
        if function == "T.Cast":
            return cast(self.eval(node.args[1]), self.eval(node.args[0]))
        if function == "T.if_then_else":
            cond = self.eval(node.args[0])
            if isinstance(cond, Expr):
                return select(cond, self.eval(node.args[1]), self.eval(node.args[2]))
            with self.budget.at('guard', node.lineno, condition=ast.unparse(node.args[0]), value=bool(cond)):
                return self.eval(node.args[1] if bool(cond) else node.args[2])
        if function == "T.match_buffer":
            buf = self.eval(node.args[0])
            shape = tuple(int(i) for i in self.eval(node.args[1]))
            requested = next((self.eval(k.value) for k in node.keywords if k.arg == "dtype"), "float32")
            if len(node.args) > 2:
                requested = self.eval(node.args[2])
            if shape != buf.shape or dtype(requested) != buf.dtype:
                raise Unsupported("TIR match_buffer 与测试参数 shape/dtype 不一致")
            strides = next((self.eval(k.value) for k in node.keywords if k.arg == "strides"), None)
            if strides is not None:
                strides=tuple(self.symbols.get(s,s) if isinstance(s,str) else s for s in strides)
            actual = tuple(int(i // buf.dtype.itemsize) for i in buf.values.strides)
            if strides is not None and tuple(strides) != actual:
                raise Unsupported("目前仅执行匹配的连续布局")
            return buf
        if function == "T.alloc_buffer":
            shape = tuple(int(i) for i in self.eval(node.args[0]))
            self.budget.shape(shape)
            requested = self.eval(node.args[1]) if len(node.args) > 1 else "float32"
            scope=next((self.eval(k.value) for k in node.keywords if k.arg=='scope'),'local')
            data=next((self.eval(k.value) for k in node.keywords if k.arg=='data'),None)
            requested=next((self.eval(k.value) for k in node.keywords if k.arg=='dtype'),requested)
            if any(k.arg not in {'scope','data','dtype'} for k in node.keywords):
                raise Unsupported(f'L{node.lineno}: allocation 属性尚未支持')
            if data is not None:
                if data not in self.handles:raise Unsupported('local.var 未声明初始化')
                initial=self.handles[data]
            else:initial=None
            if self.thread_active and scope.startswith('shared') and node.lineno in self.shared:
                return self.shared[node.lineno]
            buf=Buffer(np.empty(shape, dtype=dtype(requested)), initialized=False,
                          symbolic=self.symbolic, budget=self.budget, allocation_line=node.lineno)
            if initial is not None:
                for idx in np.ndindex(shape):buf.write(idx,initial,node.lineno)
            if self.thread_active and scope.startswith('shared'):
                self.shared[node.lineno]=buf
            return buf
        if function == "T.region":
            if not isinstance(node.args[0], ast.Subscript):
                raise Unsupported("T.region 需要 buffer 标量起点")
            buf, origin = self.address(node.args[0])
            sizes = tuple(int(self.eval(n)) for n in node.args[2:])
            if len(sizes) != len(buf.shape):
                raise Unsupported("T.region rank 不匹配")
            return (buf, origin, sizes)
        if function == "T.copy":
            if len(node.args)!=2 or any(k.arg!='disable_tma' or not bool(self.eval(k.value)) for k in node.keywords):
                raise Unsupported(f'L{node.lineno}: 未支持 copy 属性')
            source,target=[self.as_region(self.eval(n)) for n in node.args]
            if any(isinstance(i,Expr) for region in (source,target) for i in region[1]):
                raise Unsupported(f'L{node.lineno}: copy 的数据相关符号下标尚未支持；仍可单独执行具体输入')
            sa=[i for i,n in enumerate(source[2]) if n!=1];ta=[i for i,n in enumerate(target[2]) if n!=1]
            if tuple(source[2][i] for i in sa)!=tuple(target[2][i] for i in ta):
                raise Unsupported('copy 仅支持相同非单位维度，不能猜测 reshape/transpose')
            shape=tuple(source[2][i] for i in sa)
            # RISC-V TileLang scalar copy lowering src/op/copy.cc:390:
            # invalid source lanes become zero; invalid destination lanes skip.
            # Ordinary scalar Buffer.read/write still diagnose out-of-bounds.
            values=[]
            for idx in np.ndindex(shape):
                a=list(source[1]);b=list(target[1])
                for axis,i in zip(sa,idx):a[axis]+=i
                for axis,i in zip(ta,idx):b[axis]+=i
                a,b=tuple(a),tuple(b)
                if not all(0<=i<n for i,n in zip(b,target[0].shape)):continue
                value=(source[0].read(a,node.lineno) if all(0<=i<n for i,n in zip(a,source[0].shape)) else cast(0,str(target[0].dtype)))
                values.append((b,value))
            for idx,value in values:target[0].write(idx,value,node.lineno)
            return None
        if function == "T.fill":
            buf,origin,shape=self.as_region(self.eval(node.args[0]));value=self.eval(node.args[1])
            for index in np.ndindex(shape):
                buf.write(tuple(a+b for a,b in zip(origin,index)),value,node.lineno)
            return None
        if function == "T.assume":
            if not bool(self.eval(node.args[0])):
                raise Unsupported("测试不满足 T.assume")
            return None
        raise Unsupported(f"L{node.lineno}: 未支持调用 {function}")

    def as_region(self,value):
        if isinstance(value,Buffer):return value,(0,)*len(value.shape),value.shape
        if isinstance(value,tuple) and len(value)==3 and isinstance(value[0],Buffer):return value
        raise Unsupported('需要 buffer 或矩形 region')

    def physical(self, statements, variable, extent):
        """Cooperative SPMD with private local allocations and block shared data.

        The scheduler never ignores barriers or infers a missing thread layout.
        Thread-dependent annotated logical loops need a decoded Fragment mapping.
        """
        import copy
        if not 1<=extent<=1024:raise Unsupported('物理线程数超出 1..1024')
        shared={};threads=[];layout_cache={}
        for tid in range(extent):
            child=copy.copy(self);child.env=dict(self.env);child.env[variable]=np.int64(tid)
            child.handles=dict(self.handles);child.thread_active=True;child.shared=shared
            child.layout_cache=layout_cache
            child.tid=tid;child.thread_extent=extent;child.thread_region=self.budget.new_region() if tid==0 else threads[0][0].thread_region
            threads.append((child,child.cooperate(statements)))
        epoch=0
        while True:
            waits=[];done=[]
            for child,iterator in threads:
                with self.budget.iteration((child.thread_region,epoch),child.tid):
                    try:waits.append(next(iterator));done.append(False)
                    except StopIteration:waits.append(None);done.append(True)
            if all(done):break
            if any(done) or len(set(waits))!=1:
                raise SemanticFault('物理线程没有到达同一个 barrier',rule_id='SYNC-DIVERGENT-BARRIER',line=next((v for v in waits if v),None))
            epoch+=1
        self.env.update({k:v for k,v in threads[0][0].env.items() if isinstance(v,Buffer) and k in self.env})

    def cooperate(self,statements):
        for node in statements:
            self.budget.tick()
            if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and name(node.value.func)=='T.tvm_storage_sync':
                if len(node.value.args)!=1 or self.eval(node.value.args[0]) not in {'shared','shared.dyn'}:
                    raise Unsupported(f'L{node.lineno}: 未支持的 barrier scope')
                yield node.lineno
            elif isinstance(node,ast.With):yield from self.cooperate(node.body)
            elif isinstance(node,ast.Assign) and isinstance(node.value,ast.Call) and name(node.value.func)=='T.launch_thread':
                axis=self.eval(node.value.args[0]);extent=int(self.eval(node.value.args[1]))
                if extent!=1 or not axis.startswith('threadIdx.'):
                    raise Unsupported(f'L{node.lineno}: 多维物理线程暂未支持')
                self.assign(node.targets[0],np.int64(0),node.lineno)
            elif isinstance(node,ast.For):
                if any(k.arg=='annotations' and 'parallel_loop_layout' in ast.unparse(k.value) for k in node.iter.keywords):
                    yield from self.fragment_loop(node)
                    continue
                if name(node.iter.func)=='T.parallel':raise Unsupported(f'L{node.lineno}: 物理线程轴参与的 logical parallel 缺少布局映射')
                for value in range(*[int(self.eval(v)) for v in node.iter.args]):
                    self.assign(node.target,np.int64(value),node.lineno)
                    yield from self.cooperate(node.body)
            elif isinstance(node,ast.If):
                cond=self.eval(node.test)
                if isinstance(cond,Expr):raise Unsupported('物理线程中的符号条件未支持')
                yield from self.cooperate(node.body if bool(cond) else node.orelse)
            else:self.statement(node)

    def fragment_loop(self,node):
        from .layout import Fragment
        if self.metadata is None:raise Unsupported(f'L{node.lineno}: 缺少完整 Fragment metadata，不能猜测线程布局')
        attr=next(k.value for k in node.iter.keywords if k.arg=='annotations')
        mapping=next((v for k,v in zip(attr.keys,attr.values) if isinstance(k,ast.Constant) and k.value=='parallel_loop_layout'),None)
        if not isinstance(mapping,ast.Subscript) or not isinstance(mapping.slice,ast.Constant):
            raise Unsupported('Fragment 索引必须为常量')
        fragment=Fragment(self.metadata,mapping.slice.value)
        chain=[node];body=node.body
        while len(body)==1 and isinstance(body[0],ast.For) and name(body[0].iter.func)=='T.parallel':
            chain.append(body[0]);body=body[0].body
        if len(chain)!=2:raise Unsupported('当前 Fragment 需二层完整 logical parallel')
        ranges=[range(*[int(self.eval(v)) for v in n.iter.args]) for n in chain]
        if any(r.start!=0 or r.step!=1 or len(r)!=size for r,size in zip(ranges,fragment.sizes)):
            raise Unsupported('Fragment 输入域与 parallel 循环不一致')
        if node.lineno not in self.layout_cache:
            owners=[[] for _ in range(self.thread_extent)]
            for i,j in itertools.product(*ranges):
                owner=fragment.owner(i,j)
                if not 0<=owner<self.thread_extent:raise Unsupported('Fragment 映射超出物理线程组')
                owners[owner].append((i,j))
            self.layout_cache[node.lineno]=owners
        for i,j in self.layout_cache[node.lineno][self.tid]:
            for n,value in zip(chain,[i,j]):self.assign(n.target,np.int64(value),n.lineno)
            yield from self.cooperate(body)

    def reduction_region(self, node):
        """Keep singleton scalar indices as unit axes, as saved T.reduce does."""
        if isinstance(node, ast.Name):
            buf = self.eval(node)
            if not isinstance(buf, Buffer):
                raise Unsupported(f'L{node.lineno}: 归约参数不是 buffer')
            return buf, (0,) * len(buf.shape), buf.shape
        if not isinstance(node, ast.Subscript):
            raise Unsupported(f'L{node.lineno}: 不支持的归约区域')
        buf = self.eval(node.value)
        if not isinstance(buf, Buffer):
            raise Unsupported(f'L{node.lineno}: 归约参数不是 buffer')
        axes = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if len(axes) != len(buf.shape):
            raise Unsupported(f'L{node.lineno}: 归约区域 rank 不同')
        starts, sizes = [], []
        for a, bound in zip(axes, buf.shape):
            if isinstance(a, ast.Slice):
                start = int(self.eval(a.lower)) if a.lower is not None else 0
                stop = int(self.eval(a.upper)) if a.upper is not None else bound
                step = int(self.eval(a.step)) if a.step is not None else 1
                if step != 1 or stop < start:
                    raise Unsupported(f'L{node.lineno}: 归约切片只支持正向单位步长')
                starts.append(start)
                sizes.append(stop - start)
            else:
                starts.append(int(self.eval(a)))
                sizes.append(1)
        return buf, tuple(starts), tuple(sizes)

    def reduce(self, node):
        if len(node.args) != 5 or node.keywords:
            raise Unsupported(f'L{node.lineno}: 需要五参数 T.reduce')
        source, origin, shape = self.reduction_region(node.args[0])
        target, dest, dest_shape = self.reduction_region(node.args[1])
        kind, axis, clear = [self.eval(a) for a in node.args[2:]]
        if kind not in {'sum', 'max', 'min'} or not isinstance(clear, (bool, np.bool_)):
            raise Unsupported(f'L{node.lineno}: 未支持的归约种类或 clear 属性')
        if self.symbolic and kind in {'max', 'min'}:
            raise Unsupported(f'L{node.lineno}: max/min 的符号等价需补充 NaN/负零前提，当前只支持受限数值检查')
        if not isinstance(axis, (int, np.integer)):
            raise Unsupported(f'L{node.lineno}: 归约轴必须为整数')
        axis = int(axis)
        if not 0 <= axis < len(shape):
            raise Unsupported(f'L{node.lineno}: 归约轴 {axis} 超出 rank {len(shape)}')
        out_shape = shape[:axis] + shape[axis + 1:]
        if out_shape != dest_shape or source.dtype != target.dtype or source is target:
            raise Unsupported(f'L{node.lineno}: 归约输出区域/类型不匹配或存在源目标别名')
        self.reductions.append({'line': node.lineno, 'kind': kind, 'axis': axis,
            'clear': bool(clear), 'accumulator_dtype': str(target.dtype),
            'source': source.name, 'source_origin': list(origin), 'source_shape': list(shape),
            'destination': target.name, 'destination_origin': list(dest),
            'order': 'increasing_axis_index',
            'scope': 'max/min 的浮点 NaN 与负零规则未核实，数值执行遇到这些值返回 unsupported'})
        if target.dtype.kind not in 'fiu':
            raise Unsupported(f'L{node.lineno}: 不支持该归约 dtype')
        if kind == 'sum':
            initial = np.asarray(0, dtype=target.dtype)[()]
        elif target.dtype.kind == 'f':
            initial = np.asarray(-np.inf if kind == 'max' else np.inf, dtype=target.dtype)[()]
        else:
            limits = np.iinfo(target.dtype)
            initial = np.asarray(limits.min if kind == 'max' else limits.max, dtype=target.dtype)[()]
        for out_idx in np.ndindex(out_shape):
            dst = tuple(a + b for a, b in zip(dest, out_idx))
            value = initial if clear else target.read(dst, node.lineno)
            for k in range(shape[axis]):
                index = out_idx[:axis] + (k,) + out_idx[axis:]
                src = tuple(a + b for a, b in zip(origin, index))
                with self.budget.at('reduction', node.lineno, axis=axis, index=k, kind_name=kind):
                    item = source.read(src, node.lineno)
                    if not self.symbolic and kind in {'max', 'min'} and target.dtype.kind == 'f':
                        if any(np.isnan(v) or (v == 0 and np.signbit(v)) for v in [item, value]):
                            raise Unsupported(f'L{node.lineno}: max/min 的 NaN/负零规则尚未支持')
                    value = apply('add' if kind == 'sum' else kind, [value, item], str(target.dtype))
            target.write(dst, value, node.lineno)

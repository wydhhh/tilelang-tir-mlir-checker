"""Shared bounded memory/scalar model; frontends do not translate to each other."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from contextlib import contextmanager

import numpy as np


class Unsupported(ValueError):
    """Outside the explicitly implemented IR semantics."""


class SemanticFault(ValueError):
    """Concrete invalid memory/coverage behavior for a supplied test case."""

    def __init__(self, message, *, rule_id='MEM-CONTRACT', line=None, **details):
        super().__init__(message)
        self.rule_id, self.line, self.details = rule_id, line, details


def dtype(name):
    aliases = {"f32": "float32", "f64": "float64", "i1": "bool", "index": "int64"}
    if name in ("f8E4M3FN", "float8_e4m3fn"):
        try:
            import ml_dtypes
        except ImportError as exc:
            raise Unsupported("E4M3FN 执行需要 ml_dtypes>=0.5；不使用 uint8 近似浮点数") from exc
        return np.dtype(ml_dtypes.float8_e4m3fn)
    if name in ("bf16", "bfloat16"):
        try:
            import ml_dtypes
        except ImportError as exc:
            raise Unsupported("bf16 执行需要可选依赖 ml_dtypes>=0.5；未安装时不近似模拟") from exc
        return np.dtype(ml_dtypes.bfloat16)
    if name in ("float16", "f16"):
        return np.dtype("float16")
    name=aliases.get(name,name)
    if name.startswith("i") and name[1:].isdigit():
        name = "int" + name[1:]
    try:
        result = np.dtype(aliases.get(name, name))
    except TypeError as exc:
        raise Unsupported(f"未支持的数据类型 {name}") from exc
    if result.kind not in "biuf" or result.itemsize > 8:
        raise Unsupported(f"未支持的数据类型 {name}")
    return result


def storage_type(value):
    d = np.dtype(value)
    if d.name == "bfloat16":
        return "bf16"
    if d.name == "float8_e4m3fn":
        return "f8E4M3FN"
    return ("f" if d.kind == "f" else "i") + str(d.itemsize * 8) if d.kind != "b" else "i1"


@dataclass(frozen=True, eq=False)
class Expr:
    op: str
    args: tuple
    type: str
    trace: tuple = field(default=(), compare=False, repr=False)

    def __bool__(self):
        raise Unsupported("符号条件不能作为宿主 Python 布尔值")

    def __eq__(self,other):
        if not isinstance(other,Expr):return False
        pending=[(self,other)];seen=set()
        while pending:
            a,b=pending.pop()
            if a is b:continue
            if isinstance(a,Expr) and isinstance(b,Expr):
                pair=(id(a),id(b))
                if pair in seen:continue
                seen.add(pair)
                if a.op!=b.op or a.type!=b.type or len(a.args)!=len(b.args):return False
                pending.extend(zip(a.args,b.args))
            elif isinstance(a,Expr) or isinstance(b,Expr) or a!=b:return False
        return True

    __hash__=None

    def fingerprint(self):
        pending=[(self,False)]
        while pending:
            value,ready=pending.pop()
            if hasattr(value,'_fingerprint'):continue
            if not ready:
                pending.append((value,True))
                pending.extend((a,False) for a in value.args if isinstance(a,Expr) and not hasattr(a,'_fingerprint'))
                continue
            args=[['expr',a._fingerprint] if isinstance(a,Expr) else ['literal',a] for a in value.args]
            raw=json.dumps([value.op,value.type,args],separators=(',',':'),default=lambda x: x.item() if hasattr(x,'item') else str(x)).encode()
            object.__setattr__(value,'_fingerprint',hashlib.sha256(raw).hexdigest())
        return self._fingerprint

    def data(self):
        # Diagnostics may display shared DAGs with exponential expanded size.
        # Equality above remains exact; only the human-facing tree is bounded.
        remaining=[2048]
        def show(value,depth):
            if not isinstance(value,Expr):return value.item() if isinstance(value,np.generic) else value
            remaining[0]-=1
            if depth>=32 or remaining[0]<=0:return ['truncated',value.type,value.fingerprint()]
            return [value.op,value.type,*[show(a,depth+1) for a in value.args]]
        return show(self,0)


def merge_trace(values):
    records = list(dict.fromkeys(t for v in values if isinstance(v, Expr) for t in v.trace))
    return tuple(records if len(records) <= 64 else records[:32] + records[-32:])


def provenance(value):
    return [{'buffer': b, 'line': line, 'action': action, 'index': list(index)}
            for b, line, action, index in value.trace] if isinstance(value, Expr) else []


def trace_value(value, symbol, line, action, index=()):
    if not isinstance(value, Expr):
        return value
    extra = Expr('location', (), value.type, ((symbol, int(line), action, tuple(index)),))
    return replace(value, trace=merge_trace((value, extra)))


def literal(value, type_name=None):
    if isinstance(value, Expr):
        return value
    value = np.asarray(value)
    if value.ndim:
        raise Unsupported("仅支持标量表达式")
    if type_name is not None:
        value = value.astype(dtype(type_name))
    return Expr("constant", (value.tobytes().hex(),), storage_type(value.dtype))


def apply(op, values, type_name=None):
    symbolic = any(isinstance(v, Expr) for v in values)
    if symbolic:
        nodes = tuple(literal(v) for v in values)
        if op in {"ne", "eq", "lt", "le", "gt", "ge", "and", "or", "not"}:
            # Equality is insensitive to integer signedness, but not width.
            result_type = "i1"
        else:
            result_type = storage_type(dtype(type_name)) if type_name else nodes[0].type
        return Expr(op, nodes, result_type, merge_trace(nodes))
    operations = {"add": np.add, "sub": np.subtract, "mul": np.multiply,
                  "div": np.divide, "floordiv": np.floor_divide, "mod": np.remainder,
                  "ne": np.not_equal, "eq": np.equal, "lt": np.less, "le": np.less_equal,
                  "gt": np.greater, "ge": np.greater_equal,
                  "and": np.logical_and, "or": np.logical_or, "neg": np.negative,
                  "max": np.maximum, "min": np.minimum, "exp": np.exp, "exp2": np.exp2, "sqrt": np.sqrt, "log": np.log, "abs": np.abs, "tanh": np.tanh, "not": np.logical_not}
    if op == 'rsqrt':
        # MLIR math.rsqrt represents reciprocal square root. No algebraic
        # identity is used in symbolic comparisons.
        with np.errstate(all="ignore"):
            value = np.asarray(values[0])
            return cast(np.divide(np.asarray(1, dtype=value.dtype), np.sqrt(value)), type_name or str(value.dtype))
    if op == 'sigmoid':
        kind=type_name or str(np.asarray(values[0]).dtype)
        one=cast(1,kind)
        return apply('div', [one, apply('add', [one, apply('exp', [apply('neg',values)], kind)], kind)], kind)
    # Use exact Python integers before wrapping to the IR width. In particular,
    # int64 division/remainder must never pass through a host floating value.
    kind = dtype(type_name) if type_name else np.result_type(*values)
    if kind.kind in 'iu' and op in {'add','sub','mul','neg','floordiv','mod','rem'}:
        integers = list(map(int, values))
        a = integers[0]
        if op in {'floordiv','mod','rem'}:
            b = integers[1]
            if b == 0 or (kind.kind == 'i' and a == np.iinfo(kind).min and b == -1):
                raise SemanticFault('整数除法除零或有符号溢出', rule_id='ARITH-INVALID-DIVISION')
        if op == 'rem':
            q = (abs(a)//abs(b)) * (-1 if (a < 0) != (b < 0) else 1)
            answer = a-q*b
        else:
            answer = {'add':lambda:a+integers[1], 'sub':lambda:a-integers[1],
                      'mul':lambda:a*integers[1], 'neg':lambda:-a,
                      'floordiv':lambda:a//b, 'mod':lambda:a%b}[op]()
        return cast(answer, str(kind))
    if op in {'floordiv', 'mod', 'rem'} and not symbolic:
        a,b=values
        if int(b)==0 or (isinstance(a,np.signedinteger) and int(a)==np.iinfo(a.dtype).min and int(b)==-1):
            raise SemanticFault('整数除法除零或有符号溢出', rule_id='ARITH-INVALID-DIVISION')
    if op == 'rem':
        a,b=map(int,values)
        quotient=(abs(a)//abs(b)) * (-1 if (a<0)!=(b<0) else 1)
        return cast(a-quotient*b,type_name or str(np.asarray(values[0]).dtype))
    if op not in operations:
        raise Unsupported(f"未支持的标量运算 {op}")
    with np.errstate(all="ignore"):
        result = operations[op](*values)
    if type_name is not None:
        result = np.asarray(result, dtype=dtype(type_name))[()]
    return result


def cast(value, target):
    if isinstance(value, Expr):
        kind = storage_type(dtype(target))
        return value if value.type == kind else Expr("cast", (value,), kind, value.trace)
    kind = dtype(target)
    scalar = np.asarray(value)
    if scalar.ndim:
        raise Unsupported('类型转换仅接受标量')
    if kind.kind in 'iu' and (scalar.dtype.kind in 'biu' or type(value) is int):
        bits = kind.itemsize*8
        raw = int(value) % (1 << bits)
        if kind.kind == 'i' and raw >= 1 << (bits-1): raw -= 1 << bits
        return np.asarray(raw, dtype=kind)[()]
    with np.errstate(all='ignore'):
        return scalar.astype(kind)[()]


def integer_interpret(value, source, *, unsigned=False):
    """Interpret signless MLIR bits as signed/unsigned without changing width."""
    kind = dtype(source)
    if kind.kind not in 'biu': raise Unsupported('需要整数位类型')
    bits = 1 if kind.kind == 'b' else kind.itemsize*8
    if isinstance(value, Expr):
        # Keep the signedness choice in the expression: equal storage widths
        # alone do not make zero/sign extensions equivalent.
        return Expr('unsigned_bits' if unsigned else 'signed_bits', (value,),
                    ('uint' if unsigned else 'int')+str(max(8,bits)), value.trace)
    raw = int(value) % (1 << bits)
    return raw if unsigned or raw < 1 << (bits-1) else raw-(1 << bits)


def mlir_cast(operation, value, source, target):
    """Default-RNE float casts and explicit sign/zero extension of MLIR bits."""
    if operation in {'extui','extsi','uitofp','sitofp','index_cast','index_castui'}:
        value = integer_interpret(value, source, unsigned=operation in {'extui','uitofp','index_castui'})
    elif operation == 'trunci':
        # Truncation keeps low bits, independent of their signed interpretation.
        if dtype(target).kind == 'b':
            if isinstance(value, Expr):
                return Expr('trunc_bits', (value,), 'i1', value.trace)
            return np.bool_(int(value) & 1)
    elif operation not in {'extf','truncf'}:
        raise Unsupported(f'未支持类型转换 {operation}')
    return cast(value, target)


def mlir_cmpi(predicate, left, right, source):
    if predicate in {'eq','ne'} and any(isinstance(v,Expr) for v in (left,right)):
        return apply(predicate,[left,right])
    if predicate in {'eq','ne'}:
        # Equality compares bit patterns; unsigned also handles i1 exactly.
        signed = False
        operation = predicate
    elif predicate in {'slt','sle','sgt','sge','ult','ule','ugt','uge'}:
        signed = predicate[0] == 's'
        operation = predicate[1:]
    else:
        raise Unsupported(f'未知整数比较谓词 {predicate}')
    values = [integer_interpret(v,source,unsigned=not signed) for v in (left,right)]
    if not any(isinstance(v, Expr) for v in values):
        # Mixed Python integers near 2**63 can make NumPy select float64,
        # losing low bits. Integer comparisons must remain exact.
        a,b = values
        return np.bool_({'eq':lambda:a==b,'ne':lambda:a!=b,'lt':lambda:a<b,
                         'le':lambda:a<=b,'gt':lambda:a>b,'ge':lambda:a>=b}[operation]())
    return apply(operation, values)


def select(condition, yes, no):
    if isinstance(condition, Expr):
        a, b = literal(yes), literal(no)
        if a.type != b.type:
            raise Unsupported("选择分支类型不同")
        return Expr("select", (condition, a, b), a.type, merge_trace((condition, a, b)))
    return yes if bool(condition) else no


class Budget:
    def __init__(self, max_steps=2_000_000, max_elements=1_000_000):
        self.max_steps, self.max_elements = max_steps, max_elements
        self.steps = 0
        self.allocated_elements = 0
        self.parallel=[]
        self.region_count=0
        self.context=[]

    @contextmanager
    def at(self, kind, line, **values):
        self.context.append({'kind': kind, 'line': line, **values})
        try:
            yield
        finally:
            self.context.pop()

    def new_region(self):
        self.region_count+=1
        return self.region_count

    @contextmanager
    def iteration(self,region,index):
        self.parallel.append((region,index))
        try:
            yield
        finally:
            self.parallel.pop()

    def tick(self, n=1):
        self.steps += n
        if self.steps > self.max_steps:
            raise Unsupported(f"超出执行步数上限 {self.max_steps}")

    def shape(self, shape):
        if any(type(int(n)) is not int or int(n) < 0 for n in shape):
            raise Unsupported("非法形状")
        if np.prod(shape, dtype=object) > self.max_elements:
            raise Unsupported(f"单个缓冲区超出元素上限 {self.max_elements}")


class Buffer:
    def __init__(self, values, *, initialized, symbolic, position=None, budget=None,
                 name=None, allocation_line=None, readonly=False):
        budget=budget or Budget()
        budget.allocated_elements+=np.asarray(values).size
        if budget.allocated_elements>8*budget.max_elements:
            raise Unsupported('累计缓冲区分配超出元素预算，拒绝继续分配')
        self.values = np.array(values, copy=True, order="C")
        if not initialized:
            self.values.fill(0)
        self.dtype = self.values.dtype
        self.shape = self.values.shape
        self.budget = budget
        self.budget.shape(self.shape)
        self.initialized = np.full(self.shape, initialized, dtype=bool)
        self.writes = np.zeros(self.shape, dtype=np.int64)
        self.last_lines = np.zeros(self.shape, dtype=np.int64)
        self.symbolic = symbolic
        self.accesses={}
        self.name = name or (f'arg{position}' if position is not None else 'temporary')
        self.position, self.allocation_line, self.readonly = position, allocation_line, readonly
        self.expressions = None
        if symbolic:
            self.expressions = np.empty(self.shape, dtype=object)
            for index in np.ndindex(self.shape):
                self.expressions[index] = (Expr("input", (position, index), storage_type(self.dtype))
                                           if initialized else None)

    def subview(self, offsets, sizes, strides, dropped=()):
        if not (len(offsets)==len(sizes)==len(strides)==len(self.shape)):
            raise Unsupported("subview rank 不同")
        for a,n,k,b in zip(offsets,sizes,strides,self.shape):
            if a<0 or n<0 or k<=0 or a+n*k-k>=b and n:
                self.fault('MEM-OUT-OF-BOUNDS','subview 区域越界',0,tuple(offsets))
        if any(sizes[i]!=1 for i in dropped):
            raise Unsupported("只能降掉 size=1 的 subview 维度")
        index=tuple(slice(a,a+n*k,k) for a,n,k in zip(offsets,sizes,strides))
        result=object.__new__(Buffer)
        result.__dict__=dict(self.__dict__)
        for attr in ['values','initialized','writes','last_lines','expressions']:
            arr=getattr(self,attr)
            setattr(result,attr,None if arr is None else np.squeeze(arr[index],axis=tuple(dropped)))
        result.shape=result.values.shape
        result._parent=self
        result._offsets=tuple(offsets);result._strides=tuple(strides);result._dropped=tuple(dropped)
        return result

    def fault(self, rule_id, message, line, index, **details):
        raise SemanticFault(f'L{line}: {message}', rule_id=rule_id, line=line,
            buffer=self.name, parameter=self.position, shape=list(self.shape),
            index=list(index), allocation_line=self.allocation_line,
            context=list(self.budget.context), **details)

    def trace(self, value, line, action, index):
        return trace_value(value, self.name, line, action, index)

    def index(self, index, line, access='read'):
        if not isinstance(index, tuple):
            index = (index,)
        if len(index) != len(self.shape) or any(isinstance(i, Expr) for i in index):
            raise Unsupported(f"L{line}: 不支持的内存下标 {index}")
        index = tuple(int(i) for i in index)
        if any(i < 0 or i >= size for i, size in zip(index, self.shape)):
            self.fault('MEM-OUT-OF-BOUNDS', f'内存访问越界 {index}，shape={self.shape}', line, index, access=access)
        return index

    def read(self, index, line):
        self.budget.tick()
        index = self.index(index, line)
        self.access(index,False,line)
        if not self.initialized[index]:
            self.fault('MEM-UNINITIALIZED', f'读取未初始化数据 {index}', line, index, access='read')
        return self.trace(self.expressions[index], line, 'read', index) if self.symbolic else self.values[index]

    def write(self, index, value, line):
        self.budget.tick()
        index = self.index(index, line, 'write')
        if self.readonly:
            self.fault('MEM-READONLY-WRITE', '修改了只读输入参数', line, index, access='write')
        self.access(index,True,line)
        if self.symbolic:
            self.expressions[index] = self.trace(cast(literal(value), str(self.dtype)), line, 'write', index)
        else:
            self.values[index] = value
        self.initialized[index] = True
        self.writes[index] += 1
        self.last_lines[index] = line

    def access(self,index,write,line):
        if hasattr(self,'_parent'):
            iterator=iter(index)
            parent_index=tuple(a if i in self._dropped else a+next(iterator)*s for i,(a,s) in enumerate(zip(self._offsets,self._strides)))
            return self._parent.access(parent_index,write,line)
        for region,iteration in self.budget.parallel:
            record=self.accesses.setdefault(region,{}).setdefault(index,{'writer':None,'readers':set(), 'last_line':None})
            if (record['writer'] is not None and record['writer']!=iteration
                    or write and any(reader!=iteration for reader in record['readers'])):
                self.fault('PARALLEL-CONFLICT', f'逻辑 parallel 迭代之间存在读写/写写冲突，下标 {index}',
                    line, index, conflicting_line=record['last_line'], parallel_region=region,
                    previous_iteration=record['writer'], current_iteration=iteration)
            if write:
                record['writer']=iteration
            else:
                record['readers'].add(iteration)
            record['last_line']=line

    def summary(self):
        return {"buffer": self.name, "allocation_line": self.allocation_line,
                "shape": list(self.shape), "dtype": str(self.dtype),
                "elements": int(self.values.size),
                "unwritten": int(np.count_nonzero(self.writes == 0)),
                "multiply_written": int(np.count_nonzero(self.writes > 1)),
                "writer_lines": sorted(set(int(v) for v in self.last_lines.flat if v))}


def make_buffers(arguments, outputs, symbolic, budget, inouts=()):
    result = []
    for index, value in enumerate(arguments):
        if not isinstance(value, np.ndarray) or not value.flags.c_contiguous:
            raise Unsupported("当前执行后端要求连续 NumPy 数组参数")
        result.append(Buffer(value, initialized=index not in outputs or index in inouts, symbolic=symbolic,
                             position=index, budget=budget, readonly=index not in outputs and index not in inouts))
        result[-1].require_full_write=index not in inouts
    for i, a in enumerate(arguments):
        for b in arguments[:i]:
            if np.shares_memory(a, b):
                raise Unsupported("当前执行契约要求参数之间不别名")
    return result


def expression_digest(array):
    h = hashlib.sha256()
    for value in array.flat:
        h.update(value.fingerprint().encode() if value is not None else b"null")
        h.update(b"\n")
    return h.hexdigest()

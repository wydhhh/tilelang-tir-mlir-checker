"""Static contracts for the small set of Batch 3 calls (no execution dependencies)."""
import ast

from .capabilities import (TIR_BATCH_SCRIPT, TIR_BREAKS, TIR_CUMSUM, TIR_EXTERN, TIR_EXTERN_SIGNATURES,
                           TIR_SCALARS, TIR_SERIAL_MARKERS, TIR_TAIL_SCRIPT, TIR_TAIL_STATEMENTS, TIR_TUPLE)


SWIZZLE_ATTR = 'threadblock_swizzle_pattern'
WARP_FENCE = 'tl.warpgroup_fence_operand'


def tail_call(node):
    """Decode the printer spellings of the same TIR call; never evaluates AST."""
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='T':
        name = node.func.attr
        if name=='call_intrin' and len(node.args)>=2 and isinstance(node.args[1],ast.Constant):
            op = node.args[1].value
            if op in TIR_TAIL_STATEMENTS or op==TIR_EXTERN:
                return op,node.args[2:],constant(node.args[0])
        if name in ('call_extern','call_pure_extern'):
            return TIR_EXTERN,node.args[1:],constant(node.args[0]) if node.args and name=='call_extern' else None
        if 'T.'+name in TIR_TAIL_SCRIPT:
            return TIR_TAIL_SCRIPT['T.'+name],node.args,None
    return None


def constant(node):
    if isinstance(node,ast.Constant):
        return node.value
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):
        value=constant(node.operand)
        return -value if type(value) in (int,float) else None
    if (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
            and isinstance(node.func.value,ast.Name) and node.func.value.id=='T'
            and 'T.'+node.func.attr in TIR_SCALARS and len(node.args)==1 and not node.keywords):
        value=constant(node.args[0]);kind=node.func.attr
        if kind=='bool' and type(value) in (int,bool) and value in (0,1): return bool(value)
        if kind.startswith(('uint','int')) and type(value) is int:
            bits=int(kind.removeprefix('uint').removeprefix('int'))
            low=0 if kind.startswith('u') else -(1 << (bits-1))
            high=(1 << (bits if kind.startswith('u') else bits-1))-1
            if low<=value<=high: return value
        if kind.startswith(('float','bfloat')) and type(value) in (float,int):
            try: return float(value)
            except OverflowError: return None
    return None


def marker_error(op, args, *, pointer=False):
    if op in {'tir.ptx_wait_group','tl.wait_wgmma'}:
        valid = len(args)==1 and type(args[0]) is int and 0<=args[0]<=7
    elif op=='tl.set_max_nreg':
        valid = (len(args)==2 and type(args[0]) is int and 24<=args[0]<=256
                 and args[0]%8==0 and type(args[1]) in (int,bool) and args[1] in (0,1))
    elif op==WARP_FENCE:
        valid = (len(args)==4 and pointer and isinstance(args[0],str) and
                 'T.'+args[0] in TIR_SCALARS and all(type(x) is int for x in args[2:])
                 and args[2]>=0 and args[3]>0)
    else:
        valid = not args
    return None if valid else f'{op}: 参数必须为合法的静态标量；仅支持明确的串行提示签名'


def tuple_error(values, metadata=False):
    if not all(type(v) in (str,int,float,bool) for v in values):
        return 'tir.tvm_tuple 仅支持无副作用的标量字面量占位或元数据'
    if metadata and (len(values)!=2 or type(values[0]) is not str or type(values[1]) is not int or values[1]<=0):
        return 'tir.tvm_tuple: swizzle 元数据需要 (device_func, 正整数 panel_size)'
    return None


def extern_error(args, result, kinds, *, allow_unknown=False):
    symbol = args[0] if args and type(args[0]) is str else None
    sig = TIR_EXTERN_SIGNATURES.get(symbol)
    if sig is None or sig[0]!=result or len(args)!=len(sig[1])+1:
        return f'tir.call_extern {symbol!r}: 未支持的字面量符号、参数数量或返回 dtype'
    if any(got!=expected and not (allow_unknown and got is None) for got,expected in zip(kinds,sig[1])):
        return f'tir.call_extern {symbol}: 参数 dtype 必须是 float32'
    return None


def script_tail_issues(function, *, object_backend=False):
    """Check all branches and infer the scalar dtypes of supported text operands."""
    issues=[]
    types={}

    def tname(node):
        if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name) and node.value.id=='T':
            return node.attr
        return ''

    def type_string(node):
        value=constant(node)
        return value if isinstance(value,str) else None

    def declare(node):
        if isinstance(node,ast.Call) and tname(node.func) in ('Buffer','Tensor','match_buffer','alloc_buffer'):
            index=2 if tname(node.func)=='match_buffer' else 1
            kind=next((type_string(k.value) for k in node.keywords if k.arg=='dtype'),
                      type_string(node.args[index]) if len(node.args)>index else 'float32')
            return ('buffer',kind) if kind else None
        kind=tname(node)
        return kind if 'T.'+kind in TIR_SCALARS else None

    for arg in function.args.posonlyargs+function.args.args:
        types[arg.arg]=declare(arg.annotation)

    def infer(node):
        if isinstance(node,ast.Constant):
            return {int:'int64',float:'float32',bool:'bool'}.get(type(node.value))
        if isinstance(node,ast.Name):
            return types.get(node.id)
        if isinstance(node,ast.Subscript):
            kind=infer(node.value)
            return kind[1] if isinstance(kind,tuple) else None
        if isinstance(node,ast.UnaryOp):
            return 'bool' if isinstance(node.op,ast.Not) else infer(node.operand)
        if isinstance(node,(ast.Compare,ast.BoolOp)):
            return 'bool'
        if isinstance(node,ast.BinOp):
            a,b=infer(node.left),infer(node.right)
            return a if isinstance(a,str) and a==b and not (
                isinstance(node.op,ast.Div) and a.startswith(('int','uint'))) else None
        if isinstance(node,ast.Call):
            kind=tname(node.func)
            if kind=='Cast' or kind=='call_intrin' or kind=='call_extern':
                return type_string(node.args[0]) if node.args else None
            if kind=='reinterpret':
                if len(node.args)==2: return type_string(node.args[1]) or type_string(node.args[0])
                return next((type_string(k.value) for k in node.keywords if k.arg=='dtype'),None)
            if kind=='infinity':
                return type_string(node.args[0]) if node.args else None
            if kind in {'isnan','isfinite'}:
                return 'bool'
            if ('T.'+kind in TIR_BATCH_SCRIPT or kind in {'exp','exp2','rsqrt','sqrt','sigmoid'}) and node.args:
                return infer(node.args[0])
            if kind=='if_then_else' and len(node.args)==3:
                a,b=map(infer,node.args[1:])
                return a if a==b else None
            return declare(node) or (kind if 'T.'+kind in TIR_SCALARS else None)
        return None

    def check(node, loop=None, standalone=False, metadata=False):
        if isinstance(node,ast.Call):
            tail=tail_call(node)
            if tail:
                op,args,result=tail
                vals=[constant(a) for a in args]
                msg=None
                if node.keywords:
                    msg=f'{op}: 不支持附加参数'
                elif op==TIR_EXTERN:
                    msg=extern_error(vals,result,[infer(a) for a in args[1:]],allow_unknown=object_backend)
                elif not (standalone or metadata and op==TIR_TUPLE):
                    msg=f'{op}: 仅支持单独 Evaluate 或允许的元数据，不能用作值'
                elif op in TIR_BREAKS and (args or loop not in ('serial','range')):
                    msg=f'{op}: 只允许无参数地跳出最内层串行 For'
                elif op==TIR_CUMSUM and (len(args)!=4 or type(vals[2]) is not int or
                                        type(vals[3]) not in (bool,int) or vals[3] not in (0,1)):
                    msg='cumsum 需要 4 参数和静态整数 dim / 布尔 reverse'
                elif op in TIR_SERIAL_MARKERS:
                    pointer=(len(args)==4 and isinstance(args[1],ast.Attribute) and
                             args[1].attr=='data' and isinstance(infer(args[1].value),tuple))
                    if object_backend and len(args)==4 and isinstance(args[1],ast.Name):
                        pointer=True # The object visitor checks the actual handle dtype.
                    msg=marker_error(op,vals,pointer=pointer)
                elif op==TIR_TUPLE:
                    msg=tuple_error(vals,metadata)
                if msg: issues.append((node,op,msg))
            if standalone and tname(node.func)=='evaluate':
                if len(node.args)==1 and not node.keywords:
                    check(node.args[0],loop,standalone=True)
                    return
            if standalone and tname(node.func)=='attr' and len(node.args)==3 and constant(node.args[1])==SWIZZLE_ATTR:
                check(node.args[0],loop)
                check(node.args[2],loop,metadata=True)
                return
        for child in ast.iter_child_nodes(node):
            check(child,loop)

    def walk(statements,loop):
        for stmt in statements:
            if isinstance(stmt,ast.For):
                check(stmt.iter,loop)
                old=dict(types)
                if isinstance(stmt.target,ast.Name): types[stmt.target.id]='int64'
                kind=tname(stmt.iter.func) if isinstance(stmt.iter,ast.Call) else ''
                if isinstance(stmt.iter,ast.Call) and isinstance(stmt.iter.func,ast.Name) and stmt.iter.func.id=='range':
                    kind='range'
                walk(stmt.body,kind)
                types.clear();types.update(old)
                walk(stmt.orelse,loop)
            elif isinstance(stmt,ast.If):
                check(stmt.test,loop)
                old=dict(types)
                walk(stmt.body,loop)
                yes=dict(types)
                types.clear();types.update(old)
                walk(stmt.orelse,loop)
                no=dict(types)
                types.clear()
                types.update({k:yes[k] for k in yes if k in no and yes[k]==no[k]})
            elif isinstance(stmt,ast.With):
                for item in stmt.items: check(item.context_expr,loop)
                walk(stmt.body,loop)
            elif isinstance(stmt,ast.Expr):
                check(stmt.value,loop,standalone=True)
            else:
                check(stmt,loop)
                if isinstance(stmt,ast.Assign):
                    for dst in stmt.targets:
                        if isinstance(dst,ast.Name): types[dst.id]=infer(stmt.value)
                if isinstance(stmt,ast.AnnAssign) and isinstance(stmt.target,ast.Name):
                    types[stmt.target.id]=declare(stmt.annotation)
    walk(function.body,None)
    return issues

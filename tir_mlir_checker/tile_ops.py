"""Bounded region operations and scalar matrix contraction shared by the frontends."""
from __future__ import annotations

import re

import numpy as np

from .capabilities import AFFINE_MATMUL_MAP, TIR_REDUCE_COMBINE
from .execution import (Buffer, Unsupported, apply, cast, concrete_index, mlir_cast,
                        mlir_cmpf, mlir_cmpi, scalar_op, select)


def region_descriptor(buffer, origin, shape):
    origin, shape = tuple(map(concrete_index, origin)), tuple(map(concrete_index, shape))
    if len(origin) != len(buffer.shape) or len(shape) != len(buffer.shape):
        raise Unsupported('TileOp region rank 与 buffer 不同')
    buffer.budget.shape(shape)
    return buffer, origin, shape


def fill_region(region, value, line, write):
    dst, origin, shape = region_descriptor(*region)
    value = cast(value, dst.dtype.name)
    for index in np.ndindex(shape):
        write(dst, tuple(a+b for a,b in zip(origin,index)), value, line)


def reduce_region(source, target, kind, dim, clear, line, write, record):
    """LowerTileReduce: visit lanes in order and use the exact accumulator rule."""
    src, origin, shape = region_descriptor(*source)
    dst, dest, dshape = region_descriptor(*target)
    dim = concrete_index(dim)
    axis = dim + len(shape) if dim < 0 else dim
    if not isinstance(kind,str) or kind not in TIR_REDUCE_COMBINE or not 0 <= axis < len(shape) or not shape[axis]:
        raise Unsupported('reduce 需要已知归约类型、合法轴及非空归约维')
    if not isinstance(clear,(bool,np.bool_,int,np.integer)) or clear not in (0,1):
        raise Unsupported('reduce 需要静态布尔 clear')
    clear = bool(clear)
    kept = len(dshape) == len(shape)
    out_shape = shape[:axis] + shape[axis+1:]
    if dshape != (shape[:axis]+(1,)+shape[axis+1:] if kept else out_shape):
        raise Unsupported('reduce 输出尺寸须移除归约维或保留长度 1')
    if src.root is dst.root or _overlaps(src,dst):
        raise Unsupported('reduce 输入输出不得共享存储')
    sk, dk = src.dtype, dst.dtype
    bitwise = kind in {'bitand','bitor','bitxor'}
    if bitwise and (sk.kind not in 'biu' or dk.kind not in 'biu'):
        raise Unsupported('位归约需要整数或布尔 dtype')
    floating = dk.kind == 'f' or dk.name in {'bfloat16','float8_e4m3fn'}
    src_float = sk.kind == 'f' or sk.name in {'bfloat16','float8_e4m3fn'}
    if ((not floating and dk.kind not in 'biu') or (not src_float and sk.kind not in 'biu')
            or (src_float and floating and sk != dk and sk.itemsize == dk.itemsize)):
        raise Unsupported('不支持该 reduce dtype 转换')
    dt = dk.name
    zero = cast(0, dt)
    combine = TIR_REDUCE_COMBINE[kind]
    take_abs = kind in {'abssum','absmax'}
    record({'kind':kind,'axis':axis,'clear':clear,'accumulator_dtype':dt,
            'output_keeps_reduced_dim':kept,'order':'increasing_axis_index',
            'source':src.name,'source_origin':list(origin),'source_shape':list(shape),
            'destination':dst.name,'destination_origin':list(dest)})

    def convert(value):
        if sk == dk:
            return value
        if dk.kind == 'b':
            return mlir_cmpf('une',value,cast(0,sk.name)) if src_float else apply('ne',[value,cast(0,sk.name)])
        if src_float and not floating:
            # RISC-V CastValue selects signed conversion for a floating source.
            return mlir_cast('fptosi',value,sk.name,dt)
        if not src_float and floating:
            return mlir_cast('sitofp' if sk.kind == 'i' else 'uitofp',value,sk.name,dt)
        return cast(value,dt)

    def compare(pred, a, b):
        if floating:
            return mlir_cmpf('o'+pred,a,b)
        return mlir_cmpi(('s' if dk.kind == 'i' else 'u')+pred,a,b,dt)

    for idx in np.ndindex(out_shape):
        coord = idx[:axis]+(0,)+idx[axis:] if kept else idx
        dest_idx = tuple(a+b for a,b in zip(dest,coord))
        acc = (zero if combine == 'add' else None) if clear else dst.read(dest_idx,line)
        for lane in range(shape[axis]):
            coord = idx[:axis]+(lane,)+idx[axis:]
            with src.budget.at('reduction',line,axis=axis,index=lane,kind_name=kind):
                item = convert(src.read(tuple(a+b for a,b in zip(origin,coord)),line))
                if take_abs and dk.kind not in 'bu':
                    # Lowering uses ordered < 0 + select, preserving -0 and NaN.
                    neg = apply('neg',[item],dt) if floating else apply('sub',[zero,item],dt)
                    item = select(compare('lt',item,zero),neg,item)
                if acc is None:
                    acc = item
                elif combine in {'max','min'}:
                    # Ordered compare chooses the right operand on NaN or ties.
                    acc = select(compare('gt' if combine=='max' else 'lt',acc,item),acc,item)
                elif bitwise or dk.kind == 'b':
                    acc = scalar_op(combine if bitwise else 'bitxor',[acc,item],dt)
                else:
                    acc = apply('add',[acc,item],dt)
        write(dst,dest_idx,acc,line)


def cumsum_region(source,target,dim,reverse,line,write,record):
    src,origin,shape = region_descriptor(*source)
    dst,dest,dst_shape = region_descriptor(*target)
    dim = concrete_index(dim)
    axis = dim+len(shape) if dim<0 else dim
    if shape != dst_shape or not shape or 0 in shape or not 0<=axis<len(shape):
        raise Unsupported('cumsum 需要同形非空 region 和合法 dim')
    if any(b.dtype.kind not in 'iuf' and b.dtype.name not in ('bfloat16','float8_e4m3fn')
           for b in (src,dst)):
        raise Unsupported('cumsum 只支持标量 int/uint/float')
    for buf,start,access in ((src,origin,'read'),(dst,dest,'write')):
        buf.index(start,line,access)
        buf.index(tuple(a+n-1 for a,n in zip(start,shape)),line,access)
    a,b = (buf.subview(start,shape,(1,)*len(shape)) for buf,start in ((src,origin),(dst,dest)))
    identical = (a.root is b.root and a.element_offset==b.element_offset
                 and all(n==1 or s==t for n,s,t in zip(shape,a.element_strides,b.element_strides)))
    if not identical and _overlaps(a,b):
        raise Unsupported('cumsum 输入输出 region 部分重叠或元素对应不同')
    kind = dst.dtype.name
    record({'kind':'cumsum','axis':axis,'reverse':reverse,'accumulator_dtype':kind,
            'order':'decreasing_axis_index' if reverse else 'increasing_axis_index'})
    lanes = range(shape[axis]-1,-1,-1) if reverse else range(shape[axis])
    for outer in np.ndindex(shape[:axis]+shape[axis+1:]):
        acc = None
        for lane in lanes:
            coord = outer[:axis]+(lane,)+outer[axis:]
            value = cast(src.read(tuple(a+j for a,j in zip(origin,coord)),line),kind)
            acc = value if acc is None else apply('add',[acc,value],kind)
            write(dst,tuple(b+j for b,j in zip(dest,coord)),acc,line)


def _overlaps(a, b):
    try:
        return np.shares_memory(a.values, b.values, max_work=10_000)
    except getattr(np,'exceptions',np).TooHardError as exc:
        raise Unsupported('无法在预算内排除矩阵存储重叠') from exc


def copy_region(source, target, line, write):
    src, origin, a_shape = region_descriptor(*source)
    dst, dest, b_shape = region_descriptor(*target)
    if src.root is dst.root or _overlaps(src, dst):
        raise Unsupported('TileOp copy 源目标别名尚未支持')
    if len(a_shape) == len(b_shape):
        broadcast = tuple(a == 1 or n == 1 for a,n in zip(a_shape,src.shape))
        if any(a != b and not bc for a,b,bc in zip(a_shape,b_shape,broadcast)):
            raise Unsupported('TileOp copy 区域尺寸不匹配，仅支持单位维广播')
        indexes = ((tuple(a+(0 if bc else j) for a,bc,j in zip(origin,broadcast,i)),
                    tuple(b+j for b,j in zip(dest,i))) for i in np.ndindex(b_shape))
    else:
        sa = [i for i,n in enumerate(a_shape) if n != 1]
        ta = [i for i,n in enumerate(b_shape) if n != 1]
        shape = tuple(a_shape[i] for i in sa)
        if shape != tuple(b_shape[i] for i in ta):
            raise Unsupported('TileOp copy 降 rank 后尺寸不匹配')

        def reduced_indices():
            for index in np.ndindex(shape):
                a,b = list(origin),list(dest)
                for axis,i in zip(sa,index): a[axis] += i
                for axis,i in zip(ta,index): b[axis] += i
                yield tuple(a),tuple(b)
        indexes = reduced_indices()
    zero = cast(0,dst.dtype.name)
    for a,b in indexes:
        dst.budget.tick()  # Also account for masked destination lanes.
        if not all(0 <= i < n for i,n in zip(b,dst.shape)):
            continue
        value = (cast(src.read(a,line),dst.dtype.name)
                 if all(0 <= i < n for i,n in zip(a,src.shape)) else zero)
        write(dst,b,value,line)


def check_hints(op, hints):
    """Only known, validated scheduling hints may be discarded on RISC-V."""
    copies = {'tl.tileop.async_copy','tl.tileop.tma_copy'}
    for key, value in hints.items():
        is_integer = isinstance(value,(int,np.integer,bool,np.bool_))
        if key in {'parallel_loop_layout', 'loop_layout'} and op in copies:
            if type(value).__name__ == 'Fragment':
                continue
            raise Unsupported('async_copy 的 loop_layout 必须是已识别的 TileLang Fragment')
        if key == 'coalesced_width' and op in copies:
            if concrete_index(value) > 0:
                continue
        if key == 'eviction_policy' and op in copies:
            if (is_integer and value in (0,1,2)) or (isinstance(value,str) and value in (
                    'evict_normal','evict_first','evict_last')):
                continue
        if (op in copies and is_integer and value in (0,1)
                and key in {'disable_tma', 'no_implicit_async_commit_wait',
                            'is_'+op.removeprefix('tl.tileop.')}):
            continue
        if op in GEMM_OPS and is_integer and ((key == 'is_wgmma' and value in (0,1))
                               or (key in {'is_tcgen05','use_2cta'} and value == 0)):
            continue
        raise Unsupported(f'{op}: 未支持或非法注解 {key}')


GEMM_OPS = frozenset({'tl.tileop.gemm','tl.tileop.gemm_py','tl.tileop.wgmma_gemm'})


def gemm_flag(value, name):
    if not isinstance(value,(bool,np.bool_,int,np.integer)) or value not in (0,1):
        raise Unsupported(f'GEMM {name} 需要确定的布尔值')
    return bool(value)


def gemm_controls(values, regions):
    ta,tb = (gemm_flag(v,'transpose') for v in values[:2])
    dims = tuple(map(concrete_index,values[2:5]))
    if min(dims) < 0 or concrete_index(values[5]) not in (0,1,2,3):
        raise Unsupported('GEMM M/N/K 或 warp policy 不合法')
    clear = gemm_flag(values[6],'clear_accum')
    if len(values) == 16:
        stride_a,stride_b,offset_a,offset_b,k_pack,wg_wait,mbar,cy,cx = map(concrete_index,values[7:])
        if k_pack != 1 or wg_wait < -1 or mbar != 0:
            raise Unsupported('GEMM 仅支持 k_pack=1、静态 wg_wait>=-1 和无 mbar')
        # These duplicated CUDA descriptors are ignored by LowerTileGemmPy.
        # Check the frontend's concrete layout/coordinate metadata nevertheless.
        a,b,c = regions
        if (stride_a,stride_b,offset_a,offset_b,cy,cx) != (
                a[0].shape[-1],b[0].shape[-1],a[1][-1],b[1][-1],*c[1][-2:]):
            raise Unsupported('GEMM 尾部 stride/offset/C 坐标与 region 不符')
    return ta,tb,dims,clear


def gemm_view(region, static_ones=None):
    buf, origin, shape = region_descriptor(*region)
    dropped = () if len(shape) == 2 else tuple(static_ones or ())
    if len(shape)-len(dropped) != 2:
        raise Unsupported('GEMM 仅支持逻辑二维 region；只可降掉静态 1 维')
    return buf.subview(origin,shape,(1,)*len(shape),dropped)


def matmul_shape(a, b, c, ta, tb, dims=None):
    if not all(isinstance(x, Buffer) and len(x.shape)==2 for x in (a,b,c)):
        raise Unsupported('matmul 需要三个二维 buffer')
    m,k = a.shape[::-1] if ta else a.shape
    kb,n = b.shape[::-1] if tb else b.shape
    if kb != k or c.shape != (m,n) or (dims is not None and dims != (m,n,k)):
        raise Unsupported('matmul A/B/C 形状或 M/N/K 不匹配')
    if _overlaps(c,a) or _overlaps(c,b):
        raise Unsupported('matmul C 与 A 或 B 共享存储')
    return m,n,k


def accumulate_matrix(a,b,c,ta,tb,shape,line,step,write):
    m,n,k = shape
    for i in range(m):
        for j in range(n):
            acc = c.read((i,j),line)
            for r in range(k):
                with c.budget.at('reduction',line,kind_name='gemm',index=r):
                    av = a.read((r,i) if ta else (i,r),line)
                    bv = b.read((j,r) if tb else (r,j),line)
                    acc = step(av,bv,acc)
            write(c,(i,j),acc,line)


def matrix_step(a,b,c, *, unsigned=None, kinds=None):
    """Scalar body for TIR GEMM and the named-op MLIR text shorthand."""
    ak,bk,ck = kinds or (a.dtype.name,b.dtype.name,c.dtype.name)
    target = c.dtype

    def converter(source, name):
        sk, tk = source.kind, target.kind
        if sk in 'iu' and tk in 'iu':
            if source.itemsize < target.itemsize:
                op = 'extui' if (sk == 'u' if unsigned is None else unsigned) else 'extsi'
                return lambda v: mlir_cast(op,v,name,ck)
            return lambda v: cast(v,ck)
        if sk in 'iu' and (tk == 'f' or target.name == 'bfloat16'):
            op = 'uitofp' if (sk == 'u' if unsigned is None else unsigned) else 'sitofp'
            return lambda v: mlir_cast(op,v,name,ck)
        if (sk == 'f' or source.name == 'bfloat16') and (tk == 'f' or target.name == 'bfloat16'):
            return lambda v: cast(v,ck)
        raise Unsupported('matmul 标量转换尚未支持该 dtype 组合')

    ca,cb = converter(a.dtype,ak),converter(b.dtype,bk)

    def step(av,bv,acc):
        return apply('add',[acc,apply('mul',[ca(av),cb(bv)],ck)],ck)
    return step


def reduction_details(c,ta,tb,clear):
    return {'kind':'gemm','transpose_a':ta,'transpose_b':tb,'clear_accum':clear,
            'order':'increasing_k','accumulator_dtype':c.dtype.name}


def parse_matrix_map(text):
    match = re.fullmatch(AFFINE_MATMUL_MAP,text.strip())
    if not match:
        raise Unsupported('matmul 仅支持 (d0, d1, d2) 的无符号、二维投影 affine map')
    return match.groups()


def matrix_transposes(op, maps=None):
    defaults = (op == 'linalg.matmul_transpose_a', op == 'linalg.matmul_transpose_b')
    if maps is None:
        return defaults
    valid = {((('d2','d0') if ta else ('d0','d2')),
              (('d1','d2') if tb else ('d2','d1')), ('d0','d1')): (ta,tb)
             for ta in (False,True) for tb in (False,True)}
    if maps not in valid or (op != 'linalg.matmul' and valid[maps] != defaults):
        raise Unsupported('不支持该 matmul indexing_maps（只接受普通及三种标准转置）')
    return valid[maps]

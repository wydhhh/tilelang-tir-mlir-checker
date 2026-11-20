"""Semantic checks for the Batch 2 region, matrix, and shared-storage contracts."""
import importlib

import numpy as np
import pytest

from tir_mlir_checker.execution import Buffer, Budget, SemanticFault, Unsupported, expression_digest, make_buffers
from tir_mlir_checker.mlir_interpreter import MLIRInterpreter
from tir_mlir_checker.rules import mlir_inventory, tir_inventory
from tir_mlir_checker.tir_interpreter import TIRInterpreter


def tir(args, body):
    signature = ', '.join(f'{name}: T.Buffer({shape!r}, "{kind}")' for name,shape,kind in args)
    return 'class Module:\n    @T.prim_func\n' + f'    def kernel({signature}):\n' + '\n'.join('        '+s for s in body.splitlines()) + '\n'


def mlir(args, body):
    signature = ', '.join(f'%{name}: {kind}' for name,kind in args)
    return 'module {\n  func.func @kernel(' + signature + ') {\n' + body + '\n    return\n  }\n}\n'


def execute(engine, *arrays, inout=False, symbolic=False, outputs=None):
    outputs = (len(arrays)-1,) if outputs is None else outputs
    storage = make_buffers(arrays, outputs, symbolic, Budget(), outputs if inout else ())
    report = engine.run(storage,symbolic=symbolic)
    return storage, report


@pytest.mark.parametrize('dtype,given,value', [('int32','T.int32(-7)',-7), ('float16','T.float32(-2.5)',-2.5),
                                               ('int32','T.float32(3.75)',3)])
def test_fill_region_cast_and_symbolic(dtype,given,value):
    source=tir([('y',(5,),dtype)],f'T.call_intrin("handle", "tl.tileop.fill", T.region(y[1], 2, 3), {given})')
    assert not tir_inventory(source,'00_tir_initial')[1]
    engine=TIRInterpreter(source)
    (buf,),_=execute(engine,np.zeros(5,dtype=dtype))
    np.testing.assert_array_equal(buf.writes,[0,1,1,1,0])
    np.testing.assert_array_equal(buf.values,[0,value,value,value,0])
    assert engine.writer_locations[(0,(3,))]['stage_text_line']==4
    (sym,),_=execute(engine,np.zeros(5,dtype=dtype),symbolic=True)
    assert sym.expressions[1]==sym.expressions[3]
    assert sym.expressions[0] is None and sym.expressions[1].op=='constant'
    assert len(expression_digest(sym.expressions))==64


@pytest.mark.parametrize('region,outputs,rule', [
    ('y[T.Ramp(2, 1, 3)]',(0,),'MEM-OUT-OF-BOUNDS'),
    ('y[1:3]',(),'MEM-READONLY-WRITE'),
])
def test_fill_faults(region,outputs,rule):
    source=tir([('y',(4,),'int32')],f'T.fill({region}, 5)')
    with pytest.raises(SemanticFault) as exc:
        execute(TIRInterpreter(source),np.zeros(4,dtype=np.int32),outputs=outputs)
    assert exc.value.rule_id==rule


def test_fill_explicit_region_requires_matching_ramp_lanes():
    args=[('y',(4,),'int32')]
    body='T.fill(T.region(y[T.Ramp(1, 1, 2)], 2, 2), -4)'
    (buf,),_=execute(TIRInterpreter(tir(args,body)),np.zeros(4,dtype=np.int32))
    np.testing.assert_array_equal(buf.values,[0,-4,-4,0])
    with pytest.raises(Unsupported,match='Ramp lanes'):
        execute(TIRInterpreter(tir(args,body.replace('], 2, 2)','], 2, 3)'))),np.zeros(4,dtype=np.int32))


@pytest.mark.parametrize('operation,hints', [('async_copy',', coalesced_width=4'),('tma_copy',', eviction_policy="evict_last"')])
def test_serial_copy_tail_and_alias(operation,hints):
    args=[('x',(3,),'int32'),('y',(3,),'float16')]
    body=f'T.{operation}(T.region(x[1], 1, 5), T.region(y[0], 2, 5){hints})'
    (x,y),_=execute(TIRInterpreter(tir(args,body)),np.arange(3,dtype=np.int32),np.empty(3,dtype=np.float16))
    np.testing.assert_array_equal(y.values,[1,2,0])
    np.testing.assert_array_equal(y.writes,[1,1,1])
    np.testing.assert_array_equal(x.writes,[0,0,0])
    same=f'T.{operation}(T.region(y[0], 1, 1), T.region(y[2], 2, 1){hints})'
    with pytest.raises(Unsupported,match='别名'):
        execute(TIRInterpreter(tir(args,same)),np.arange(3,dtype=np.int32),np.empty(3,dtype=np.float16))


def test_serial_copy_scalar_broadcast_and_hint_rejection():
    args=[('a',(1,2),'int32'),('c',(3,2),'int32')]
    a,c=np.array([[7,8]],dtype=np.int32),np.empty((3,2),dtype=np.int32)
    (x,y),_=execute(TIRInterpreter(tir(args,'T.async_copy(a, c)')),a,c)
    np.testing.assert_array_equal(y.values,[[7,8]]*3)
    for body in ('T.async_copy(a, c, 7)','T.async_copy(a, c, coalesced_width=0)',
                 'T.async_copy(a, c, annotations={"semantic_mask": 1})',
                 'T.tma_copy(a, c, eviction_policy=3)', 'T.tma_copy(a, c, barrier=a[0, 0])'):
        with pytest.raises(Unsupported):
            execute(TIRInterpreter(tir(args,body)),a,c)


def matrix_type(shape, kind):
    return 'memref<'+'x'.join(map(str,shape))+'x'+kind+'>'


def maps(ta,tb):
    return ('indexing_maps = ['+', '.join(f'affine_map<(d0, d1, d2) -> ({v})>'
             for v in ('d2, d0' if ta else 'd0, d2','d1, d2' if tb else 'd2, d1','d0, d1'))+']')


def matrix_program(a,b,c,t_in,t_out,ta=False,tb=False,initialize=False,custom=False,unsigned=False):
    ats,bts,cts=(matrix_type(x.shape,k) for x,k in zip((a,b,c),(t_in,t_in,t_out)))
    args=[('a',ats),('b',bts),('c',cts)]
    op=('linalg.matmul '+maps(ta,tb) if custom else
        'linalg.matmul_transpose_a' if ta else 'linalg.matmul_transpose_b' if tb else 'linalg.matmul')
    if unsigned: op += ' cast = #linalg.type_fn<cast_unsigned>'
    body = f'    {op} ins(%a, %b : {ats}, {bts}) outs(%c : {cts})'
    if initialize:
        zero='0.0' if t_out[0]=='f' else '0'
        body = f'    %zero = arith.constant {zero} : {t_out}\n    linalg.fill ins(%zero : {t_out}) outs(%c : {cts})\n' + body
    return mlir(args,body)


@pytest.mark.parametrize('ta,tb,operation', [(False,False,'gemm'),(True,False,'gemm_py'),
                                           (False,True,'wgmma_gemm'),(True,True,'gemm')])
@pytest.mark.parametrize('kind,result,mlir_in,mlir_out',[
    ('float32','float32','f32','f32'),('float16','float32','f16','f32'),('int8','int32','i8','i32')])
def test_matmul_cross_frontend_values_and_fingerprints(ta,tb,operation,kind,result,mlir_in,mlir_out):
    a=np.array([[1,-2,3],[-4,5,1]],dtype=kind)
    b=np.array([[2,-1,5,3],[3,2,-1,-4],[-3,4,1,2]],dtype=kind)
    c=np.arange(8,dtype=result).reshape((2,4))
    expected=c+a.astype(result)@b.astype(result)
    a=np.ascontiguousarray(a.T if ta else a); b=np.ascontiguousarray(b.T if tb else b)
    args=[('a',a.shape,kind),('b',b.shape,kind),('c',c.shape,result)]
    tir_engine=TIRInterpreter(tir(args,f'T.{operation}(a, b, c, transpose_A={ta}, transpose_B={tb})'))
    for custom in (False,True) if not (ta and tb) else (True,):
        mlir_source=matrix_program(a,b,c,mlir_in,mlir_out,ta,tb,custom=custom)
        assert not mlir_inventory(mlir_source,'03_mlir_linalg',{'events':[]})[1]
        mlir_engine=MLIRInterpreter(mlir_source)
        digests=[]
        for engine in (tir_engine,mlir_engine):
            result_set,report=execute(engine,a,b,c,inout=True)
            np.testing.assert_array_equal(result_set[-1].values,expected)
            assert report['reductions'][0].items() >= {
                'kind':'gemm','transpose_a':ta,'transpose_b':tb,'clear_accum':False,
                'order':'increasing_k','accumulator_dtype':result}.items()
            syms,_=execute(engine,a,b,c,inout=True,symbolic=True)
            digests.append(expression_digest(syms[-1].expressions))
        assert digests[0]==digests[1]


def test_gemm_reads_c_and_dynamic_clear():
    a=np.array([[2,-3]],dtype=np.float32)
    b=np.array([[5],[-2]],dtype=np.float32)
    c=np.empty((1,1),dtype=np.float32)
    args=[('a',a.shape,'float32'),('b',b.shape,'float32'),('c',c.shape,'float32')]
    for engine in (TIRInterpreter(tir(args,'T.gemm(a, b, c)')),
                   MLIRInterpreter(matrix_program(a,b,c,'f32','f32'))):
        with pytest.raises(SemanticFault) as exc:
            execute(engine,a,b,c)
        assert exc.value.rule_id=='MEM-UNINITIALIZED'
    for engine in (TIRInterpreter(tir(args,'T.gemm(a, b, c, clear_accum=True)')),
                   MLIRInterpreter(matrix_program(a,b,c,'f32','f32',initialize=True))):
        result,_=execute(engine,a,b,c)
        assert result[-1].values[0,0]==16
        assert result[-1].writes[0,0]==2
    dynamical=TIRInterpreter(tir(args,'for k in T.serial(2):\n    T.wgmma_gemm(a[:, k:k+1], b[k:k+1, :], c, clear_accum=k == 0)'))
    result,report=execute(dynamical,a,b,c)
    assert result[-1].values[0,0]==16
    assert [i['clear_accum'] for i in report['reductions']]==[True,False]
    assert result[-1].writes[0,0]==3


def test_raw_gemm_full_signature_and_unsupported_modes():
    a,b,c=(np.arange(4,dtype=np.int32).reshape((2,2))+1 for _ in range(3))
    args=[(name,(2,2),'int32') for name in 'abc']
    head='T.call_intrin("handle", "tl.tileop.wgmma_gemm", a, b, c, False, False, 2, 2, 2, 0, True, '
    source=tir(args,head+'2, 2, 0, 0, 1, -1, 0, 0, 0)')
    result,_=execute(TIRInterpreter(source),a,b,c)
    np.testing.assert_array_equal(result[-1].values,a@b)
    for bad in ('2, 2, 0, 0, 2, -1, 0, 0, 0)', '2, 2, 0, 0, 1, -2, 0, 0, 0)',
                '2, 2, 0, 0, 1, -1, 1, 0, 0)', '9, 2, 0, 0, 1, 0, 0, 0, 0)'):
        with pytest.raises(Unsupported,match='GEMM'):
            execute(TIRInterpreter(tir(args,head+bad)),a,b,c)


def test_gemm_logical_rank_checks_alias_and_dimensions():
    a=np.arange(12,dtype=np.int32).reshape(2,2,3)
    b=np.arange(6,dtype=np.int32).reshape(3,2)
    c=np.empty((2,2),np.int32)
    args=[('a',a.shape,'int32'),('b',b.shape,'int32'),('c',c.shape,'int32')]
    engine=TIRInterpreter(tir(args,'T.gemm(T.region(a[1, 0, 0], 1, 1, 2, 3), b, c, clear_accum=True)'))
    result,_=execute(engine,a,b,c)
    np.testing.assert_array_equal(result[-1].values,a[1]@b)
    singleton=np.ascontiguousarray(a[1:2])
    batched_args=[('a',singleton.shape,'int32'),*args[1:]]
    result,_=execute(TIRInterpreter(tir(batched_args,'T.gemm(a, b, c, clear_accum=True)')),singleton,b,c)
    np.testing.assert_array_equal(result[-1].values,a[1]@b)
    for body in ('T.gemm(a[1, :, :2], b, c, clear_accum=True)',
                 'T.gemm(c, c, c, clear_accum=True)'):
        with pytest.raises(Unsupported,match='形状|共享'):
            execute(TIRInterpreter(tir(args,body)),a,b,c)
    name='memref<2x2xi32>'
    bad=mlir([('a',name),('b',name)],f'    linalg.matmul ins(%a, %b : {name}, {name}) outs(%a : {name})')
    with pytest.raises(Unsupported,match='共享'):
        execute(MLIRInterpreter(bad),c,c.copy(),inout=True)


def test_matmul_overlap_is_physical_and_order_is_increasing():
    args=[('x','memref<8xi32>'),('b','memref<2x2xi32>')]

    def matmul_views(offset):
        a_type='memref<2x2xi32>'
        c_type=f'memref<2x2xi32, strided<[2, 1], offset: {offset}>>'
        return mlir(args,f'''    %a = memref.reinterpret_cast %x to offset: [0], sizes: [2, 2], strides: [2, 1] : memref<8xi32> to {a_type}
    %c = memref.reinterpret_cast %x to offset: [{offset}], sizes: [2, 2], strides: [2, 1] : memref<8xi32> to {c_type}
    linalg.matmul ins(%a, %b : {a_type}, {a_type}) outs(%c : {c_type})''')

    x,b=np.arange(8,dtype=np.int32),np.eye(2,dtype=np.int32)
    with pytest.raises(Unsupported,match='共享存储'):
        execute(MLIRInterpreter(matmul_views(3)),x,b,inout=True,outputs=(0,))
    (storage,_),_=execute(MLIRInterpreter(matmul_views(4)),x,b,inout=True,outputs=(0,))
    np.testing.assert_array_equal(storage.values,np.r_[x[:4],x[:4]+x[4:]])
    a=np.array([[2**24,1,-2**24]],np.float32); b=np.ones((3,1),np.float32); c=np.zeros((1,1),np.float32)
    matrices=[(n,m.shape,'float32') for n,m in zip('abc',(a,b,c))]
    for engine in (TIRInterpreter(tir(matrices,'T.gemm(a, b, c)')),
                   MLIRInterpreter(matrix_program(a,b,c,'f32','f32'))):
        result,_=execute(engine,a,b,c,inout=True)
        assert result[-1].values[0,0]==0  # Grouping the canceling terms first gives 1.


def test_matmul_index_maps_and_integer_cast_contract():
    a=np.array([[-1]],np.int8); b=np.array([[2]],np.int8); c=np.array([[1]],np.int32)
    source=matrix_program(a,b,c,'i8','i32',custom=True,unsigned=True)
    result,_=execute(MLIRInterpreter(source),a,b,c,inout=True)
    assert result[-1].values[0,0]==511
    bad=source.replace('(d0, d2)>','(d0 + 1, d2)>',1)
    with pytest.raises(Unsupported,match='indexing_maps'):
        MLIRInterpreter(bad)
    bad=source.replace('(d0, d1)>','(d1, d0)>',1)
    with pytest.raises(Unsupported,match='indexing_maps'):
        MLIRInterpreter(bad)
    with pytest.raises(Unsupported,match='属性'):
        MLIRInterpreter(source.replace('cast =','mystery =',1))
    alias='#ma = affine_map<(d0, d1, d2) -> (d0, d2)>'
    replacement=source.replace(alias.split(' = ')[1],'#ma',1)
    another,_=execute(MLIRInterpreter(alias+'\n'+replacement),a,b,c,inout=True)
    np.testing.assert_array_equal(result[-1].values,another[-1].values)


def test_reinterpret_root_storage_symbolic_and_coverage():
    root=Buffer(np.empty((3,4),np.int32),initialized=False,symbolic=True)
    sub=root.subview((1,0),(2,4),(1,1))
    view=sub.reinterpret_cast(1,(2,2),(4,2),element_type='i32')
    assert view.root is root and view.root_index((1,1))==(1,3)
    with pytest.raises(SemanticFault) as exc:
        view.read((1,1),7)
    assert exc.value.rule_id=='MEM-UNINITIALIZED'
    view.write((1,1),77,19)
    assert root.read((1,3),21)==view.read((1,1),22)
    assert (root.writes[1,3],sub.writes[0,3],view.writes[1,1])==(1,1,1)
    assert root.last_lines[1,3]==19
    for name in ('values','initialized','writes','last_lines','expressions'):
        assert np.shares_memory(getattr(root,name),getattr(view,name))
    empty=root.reinterpret_cast(500,(0,2),(4,1))
    assert empty.shape==(0,2) and empty.element_offset==500


def reinterpret_program(*, offset='1', stride='2', kind='i32', out_type=None):
    orig='memref<12xi32>'
    view=out_type or f'memref<2x2x{kind}, strided<[4, {stride}], offset: {offset}>>'
    return mlir([('x',orig)],f'''    %v = memref.reinterpret_cast %x to offset: [{offset}], sizes: [2, 2], strides: [4, {stride}] : {orig} to {view}
    %flat = memref.reinterpret_cast %v to offset: [0], sizes: [12], strides: [1] : {view} to {orig}
    %i = arith.constant 1 : index
    %j = arith.constant 7 : index
    %datum = arith.constant 23 : i32
    memref.store %datum, %v[%i, %i] : {view}
    %v2 = memref.load %flat[%j] : {orig}
    memref.store %v2, %flat[%i] : {orig}''')


def test_reinterpret_text_two_ranks_and_nonzero_stride():
    engine=MLIRInterpreter(reinterpret_program())
    (root,),_=execute(engine,np.zeros(12,dtype=np.int32),inout=True)
    assert (root.values[1],root.values[7])==(23,23)
    assert root.writes.sum()==2 and root.writes[7]==1
    assert engine.writer_locations[(0,(7,))]['stage_text_line']==8


@pytest.mark.parametrize('args,exception,message', [
    ({'offset':'10'},SemanticFault,'根缓冲区'),
    ({'stride':'-1'},Unsupported,'正 stride'),
    ({'kind':'f32'},Unsupported,'改变元素类型'),
    ({'out_type':'memref<2x2xi32, strided<[4, 2], offset: 2>>'},Unsupported,'offset/stride'),
])
def test_reinterpret_rejects_unsafe_or_inconsistent_views(args,exception,message):
    with pytest.raises(exception,match=message) as exc:
        execute(MLIRInterpreter(reinterpret_program(**args)),np.zeros(12,dtype=np.int32),inout=True)
    if exception is SemanticFault:
        assert exc.value.rule_id=='MEM-OUT-OF-BOUNDS' and exc.value.line==3


def dynamic_view_source(parallel):
    typ='memref<?x?xi32, strided<[?, ?], offset: ?>>'
    body=f'''    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %v = arith.constant 9 : i32
    %view = memref.reinterpret_cast %x to offset: [%one], sizes: [%two, %two], strides: [%two, %one] : memref<5xi32> to {typ}'''
    if parallel:
        body+=f'''
    scf.parallel (%i) = (%zero) to (%two) step (%one) {{
      %cond = arith.cmpi eq, %i, %zero : index
      scf.if %cond {{
        memref.store %v, %view[%zero, %zero] : {typ}
      }} else {{
        memref.store %v, %x[%one] : memref<5xi32>
      }}
      scf.reduce
    }}'''
    else:
        body+=f'\n    memref.store %v, %view[%one, %one] : {typ}'
    return mlir([('x','memref<5xi32>')],body)


def test_dynamic_reinterpret_parallel_alias_and_write_locations():
    with pytest.raises(SemanticFault) as exc:
        execute(MLIRInterpreter(dynamic_view_source(True)),np.zeros(5,dtype=np.int32),inout=True)
    assert exc.value.rule_id=='PARALLEL-CONFLICT' and exc.value.details['index']==[1]
    engine=MLIRInterpreter(dynamic_view_source(False))
    (x,),_=execute(engine,np.zeros(5,dtype=np.int32),inout=True)
    np.testing.assert_array_equal(x.values,[0,0,0,0,9])
    assert (0,(4,)) in engine.writer_locations


def test_real_tvm_objects():
    tvm=pytest.importorskip('tilelang',reason='requires matching TileLang/TVM Python').tvm
    from tir_mlir_checker.tvm_interpreter import TVMInterpreter
    t=tvm.tir
    a,b,c=(t.decl_buffer((2,2),'float32',name=n) for n in 'abc')

    def region(buf,mask):
        return t.Call('handle',tvm.ir.Op.get('tl.tileop.region'),
                      [t.BufferLoad(buf,[0,0]),t.IntImm('int32',mask),t.IntImm('int32',2),t.IntImm('int32',2)])

    def tile(op,args):
        args=[t.IntImm('bool',x) if type(x) is bool else t.IntImm('int32',x) if type(x) is int else x for x in args]
        return t.Evaluate(t.Call('handle',tvm.ir.Op.get('tl.tileop.'+op),args))

    mapping={x.data:x for x in (a,b,c)}
    stmt=t.SeqStmt([tile('fill',[region(c,2),t.FloatImm('float32',1)]),
                    tile('gemm',[region(a,1),region(b,1),region(c,3),False,False,2,2,2,0,False,
                                 2,2,0,0,1,0,0,0,0])])
    engine=TVMInterpreter(t.PrimFunc(list(mapping),stmt,buffer_map=mapping))
    m=np.arange(4,dtype=np.float32).reshape(2,2)
    result,report=execute(engine,m,m.copy(),np.empty_like(m))
    np.testing.assert_array_equal(result[-1].values,m@m+1)
    assert {e['action'] for e in report['node_events']} >= {'tile_fill','tile_gemm'}
    stmt=tile('async_copy',[region(a,1),region(c,2)])
    result,_=execute(TVMInterpreter(t.PrimFunc(list(mapping),stmt,buffer_map=mapping)),
                     m,m.copy(),np.empty_like(m))
    np.testing.assert_array_equal(result[-1].values,m)


def test_real_mlir_objects():
    ir=pytest.importorskip('mlir.ir',reason='requires official MLIR Python bindings')
    from tir_mlir_checker.mlir_frontend import DIALECTS
    from tir_mlir_checker.mlir_object_interpreter import MLIRObjectInterpreter
    for name in DIALECTS:
        importlib.import_module('mlir.dialects.'+name)

    def engine(source):
        ctx=ir.Context()
        with ctx:
            ctx.load_all_available_dialects()
            mod=ir.Module.parse(source)
            assert mod.operation.verify()
        return MLIRObjectInterpreter((ctx,mod),source)

    for np_kind,kind,c_kind,ct in [('float16','f16','float32','f32'),('int8','i8','int32','i32')]:
        a=np.array([[2,-1]],dtype=np_kind); b=np.array([[4],[3]],dtype=np_kind); c=np.array([[2]],dtype=c_kind)
        source=matrix_program(a,b,c,kind,ct)
        inputs=(a,b,c)
        before,_=execute(engine(source),*inputs,inout=True)
        assert before[-1].values[0,0]==7
        if kind=='i8':
            # MLIR prints this named op property in an attribute dictionary.
            with_unsigned=source.replace('linalg.matmul ins(',
                'linalg.matmul {cast = #linalg.type_fn<cast_unsigned>} ins(')
            unsigned=engine(with_unsigned)
            with unsigned.context:
                generic=unsigned.module.operation.get_asm(print_generic_op_form=True)
            assert '"arith.extui"' in generic
            changed,_=execute(unsigned,*inputs,inout=True)
            assert changed[-1].values[0,0]==2+2*4+255*3
    (x,),_=execute(engine(dynamic_view_source(False)),np.zeros(5,dtype=np.int32),inout=True)
    assert x.values[4]==9

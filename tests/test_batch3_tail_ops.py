"""Batch 3 numerical, control-flow, and whitelist contracts."""
import numpy as np
import pytest

from tir_mlir_checker.execution import Expr, SemanticFault, Unsupported, expression_digest
from tir_mlir_checker.rules import tir_inventory
from tir_mlir_checker.tir_interpreter import TIRInterpreter
from tests.test_batch2_tile_ops import execute, tir


@pytest.mark.parametrize('axis,reverse', [(0,False),(0,True),(1,False),(-1,True)])
def test_cumsum_axes_reverse_and_dtype(axis,reverse):
    a=np.array([[1,2,-3],[4,-5,7]],dtype=np.int16)
    y=np.zeros(a.shape,np.float32)
    args=[('a',a.shape,'int16'),('y',y.shape,'float32')]
    prog=tir(args,f'T.call_intrin("handle", "tl.tileop.cumsum", a[:, :], y[:, :], {axis}, {reverse})')
    assert not tir_inventory(prog,'00_tir_initial')[1]
    (inp,out),report=execute(TIRInterpreter(prog),a,y)
    work=np.flip(a,axis) if reverse else a
    expected=work.astype('float32').cumsum(axis,dtype='float32')
    np.testing.assert_array_equal(out.values,np.flip(expected,axis) if reverse else expected)
    np.testing.assert_array_equal(out.writes,1)
    assert not inp.writes.any()
    assert report['reductions'][0].items() >= {
        'kind':'cumsum','axis':axis%2,'reverse':reverse,'accumulator_dtype':'float32'}.items()


@pytest.mark.parametrize('reverse',[False,True])
def test_cumsum_same_region_preserves_order_and_symbolic_first_element(reverse):
    a=np.array([3,2**24,1,-2**24,-0.0,4],np.float32)
    source=tir([('x',(6,),'float32')],f'T.cumsum(x[1:5], x[1:5], -1, {reverse})')
    (out,),report=execute(TIRInterpreter(source),a,inout=True)
    expected=a.copy()
    first=4 if reverse else 1
    lanes=(3,2,1) if reverse else (2,3,4)
    acc=a[first]
    for lane in lanes:
        acc=np.float32(acc+a[lane]); expected[lane]=acc
    np.testing.assert_array_equal(out.values.view('uint32'),expected.view('uint32'))
    np.testing.assert_array_equal(out.writes,[0,1,1,1,1,0])
    assert report['reductions'][0]['reverse'] is reverse
    (out,),_=execute(TIRInterpreter(source),a,inout=True,symbolic=True)
    assert out.expressions[first] == Expr('input',(0,(first,)),'f32')
    assert out.expressions[1 if reverse else 4].op=='add'
    other,_=execute(TIRInterpreter(source),a,inout=True,symbolic=True)
    assert expression_digest(out.expressions)==expression_digest(other[0].expressions)


def test_cumsum_distinct_regions_and_integer_wrap():
    source=tir([('x',(6,),'uint8')],'T.cumsum(x[T.Ramp(0, 1, 3)], x[3:6], 0, False)')
    data=np.array([255,2,3,88,88,88],np.uint8)
    (out,),_=execute(TIRInterpreter(source),data,inout=True)
    np.testing.assert_array_equal(out.values,[255,2,3,255,1,4])
    with pytest.raises(Unsupported,match='重叠'):
        execute(TIRInterpreter(source.replace('x[3:6]','x[1:4]')),data,inout=True)


@pytest.mark.parametrize('body,message', [
    ('T.cumsum(x[0:0], x[0:0], 0, False)','非空'),
    ('T.cumsum(x[0:2], x[0:3], 0, False)','同形'),
    ('T.cumsum(x, x, -2, False)','合法 dim'),
    ('T.cumsum(x, x, 0, 2)','布尔 reverse'),
    ('for ax in T.serial(1):\n    T.cumsum(x, x, ax, False)','静态'),
])
def test_bad_cumsum_contracts(body,message):
    with pytest.raises(Unsupported,match=message):
        execute(TIRInterpreter(tir([('x',(3,),'float16')],body)),np.zeros(3,np.float16),inout=True)


def test_cumsum_readonly_and_bounds_diagnostics():
    data=np.ones(3,np.float32)
    args=[('a',(3,),'float32'),('b',(3,),'float32')]
    with pytest.raises(SemanticFault) as exc:
        execute(TIRInterpreter(tir(args,'T.cumsum(a, b, 0, False)')),data,data.copy(),outputs=())
    assert exc.value.rule_id=='MEM-READONLY-WRITE'
    with pytest.raises(SemanticFault) as exc:
        execute(TIRInterpreter(tir(args,'T.cumsum(a[1:4], b, 0, False)')),data,data.copy())
    assert exc.value.rule_id=='MEM-OUT-OF-BOUNDS'


def test_break_skips_remaining_statements_and_restores_binding():
    prog=tir([('y',(4,),'int32')],'''i = T.int32(17)
for i in T.serial(3):
    if i == 1:
        T.evaluate(T.loop_break())
    y[i] = i + 7
y[3] = i''')
    engine=TIRInterpreter(prog)
    (out,),_=execute(engine,np.zeros(4,np.int32))
    np.testing.assert_array_equal(out.values,[7,0,0,17])
    np.testing.assert_array_equal(out.writes,[1,0,0,1])
    assert engine.loop_stack==[] and engine.env['i']==17


def test_break_only_inner_loop_and_raw_alias():
    prog=tir([('y',(2,4),'int32')],'''for i in T.serial(2):
    for j in T.serial(3):
        if j == 1:
            T.call_intrin("handle", "tl.break_loop")
        y[i, j] = i + 2
    y[i, 3] = i + 8''')
    (out,),_=execute(TIRInterpreter(prog),np.zeros((2,4),np.int32))
    np.testing.assert_array_equal(out.values,[[2,0,0,8],[3,0,0,9]])


@pytest.mark.parametrize('body', [
    'T.loop_break()','for i in T.parallel(2):\n    T.loop_break()',
    'for i in T.serial(2):\n    for j in T.parallel(1):\n        T.break_loop()',
    'for i in T.unroll(2):\n    T.break_loop()','for i in T.serial(1):\n    T.loop_break(1)',
])
def test_break_rejects_wrong_context_or_arity(body):
    source=tir([('y',(2,),'int32')],body)
    with pytest.raises(Unsupported,match='串行 For'):
        TIRInterpreter(source)
    assert tir_inventory(source,'00_tir_initial')[1]


def test_break_with_input_condition_is_concrete_only():
    source=tir([('a',(2,),'int32'),('y',(2,),'int32')],'''for i in T.serial(2):
    if a[i] == 0:
        T.break_loop()
    y[i] = a[i]''')
    engine=TIRInterpreter(source)
    a=np.array([0,2],np.int32)
    (inp,out),_=execute(engine,a,np.zeros_like(a))
    assert not out.writes.any()
    with pytest.raises(Unsupported,match='符号'):
        execute(engine,a,np.zeros_like(a),symbolic=True)
    assert engine.loop_stack==[] and 'i' not in engine.env


def test_serial_markers_and_tuple_are_recorded():
    source=tir([('y',(1,),'int32')],'''T.ptx_wait_group(0)
T.evaluate(T.call_intrin("handle", "tl.wait_wgmma", T.int32(3)))
T.ptx_commit_group()
T.set_max_nreg(24, 0)
T.no_set_max_nreg()
T.warpgroup_fence_operand("int32", y.data, 0, 1)
T.fence_proxy_async()
T.ptx_fence_barrier_init()
T.evaluate(T.tvm_tuple(1, "marker"))
T.attr(None, "threadblock_swizzle_pattern", T.tvm_tuple("swizzle", 4))
y[0] = 12''')
    (out,),report=execute(TIRInterpreter(source),np.zeros(1,np.int32))
    assert out.values[0]==12 and out.writes[0]==1
    assert len(report['node_events'])==10
    assert all(v['action']=='serial_noop' for v in report['node_events'])
    assert report['node_events'][-1]['operation']=='tir.tvm_tuple'


@pytest.mark.parametrize('call', [
    'T.ptx_wait_group(-1)','T.ptx_wait_group(8)','T.wait_wgmma(y[0])',
    'T.ptx_commit_group(0)','T.set_max_nreg(23, 1)',
    'T.warpgroup_fence_operand("int32", y[0], 0, 1)',
    'T.tvm_tuple(y[0])','v = T.tvm_tuple(1, 2)',
    'T.attr(None, "other_metadata", T.tvm_tuple("swizzle", 4))',
    'T.shfl_down_sync(0, 0, 0)','T.warp_reduce_sum(0)','T.sync_warp()','T.sync_grid()','T.pdl_sync()',
    'T.call_intrin("handle", "tl.mbarrier_wait_parity", 0)',
    'T.call_intrin("handle", "tir.tvm_thread_allreduce", 0)',
    'T.call_intrin("handle", "tl.tileop.finalize_reducer", 0)',
    'T.call_intrin("handle", "tl.atomic_add", 0)','T.call_intrin("handle", "tl.tileop.atomicadd", 0)',
])
def test_markers_reject_side_effects_tuple_values_and_real_sync(call):
    with pytest.raises(Unsupported):
        TIRInterpreter(tir([('y',(1,),'int32')],'if False:\n    '+call))


def test_storage_sync_unchanged_and_not_a_noop():
    with pytest.raises(Unsupported,match='barrier'):
        execute(TIRInterpreter(tir([('y',(1,),'int32')],'T.tvm_storage_sync("shared")')),np.zeros(1,np.int32))


def test_fdividef_f32_and_large_divisor():
    a=np.array([7,1,np.inf,0,-1],np.float32)
    b=np.array([-2,-np.finfo('float32').max,np.finfo('float32').max,0,2**126],np.float32)
    args=[(n,a.shape,'float32') for n in ('a','b','y')]
    source=tir(args,'''for i in T.serial(5):
    y[i] = T.call_extern("float32", "__fdividef", a[i], b[i])''')
    result,_=execute(TIRInterpreter(source),a,b,np.zeros_like(a))
    expected=np.array([-3.5,-0.0,np.nan,np.nan,-2.0**-126],np.float32)
    for i in [0,1,4]:
        assert result[-1].values[i].view('uint32')==expected[i].view('uint32')
    assert np.isnan(result[-1].values[2:4]).all()
    sym,_=execute(TIRInterpreter(source),a,b,np.zeros_like(a),symbolic=True)
    assert sym[-1].expressions[0].op=='div' and sym[-1].expressions[0].args[1].op=='select'
    assert len(expression_digest(sym[-1].expressions))==64


def test_float2half_rz_exact_boundary_bits_and_symbolic():
    a=np.array([1+3*2.0**-11,-1-3*2.0**-11,1+2.0**-11,3*2.0**-25,-3*2.0**-25,
                2.0**-25,-2.0**-25,65520,-65520,65504,-0.0,np.inf,-np.inf,np.nan],np.float32)
    expected=[0x3c01,0xbc01,0x3c00,0x0001,0x8001,0x0000,0x8000,
              0x7bff,0xfbff,0x7bff,0x8000,0x7c00,0xfc00,0x7e00]
    args=[('x',a.shape,'float32'),('y',a.shape,'float16')]
    source=tir(args,f'''for i in T.serial({len(a)}):
    y[i] = T.call_intrin("float16", "tir.call_extern", "__float2half_rz", x[i])''')
    out,_=execute(TIRInterpreter(source),a,np.empty(a.shape,np.float16))
    actual=out[-1].values.view('uint16')
    np.testing.assert_array_equal(actual,expected)
    with np.errstate(over='ignore'):
        rne=a.astype('float16').view('uint16')
    assert list(np.flatnonzero(rne!=actual))==[0,1,3,4,7,8]
    sym,_=execute(TIRInterpreter(source),a,np.empty(a.shape,np.float16),symbolic=True)
    assert sym[-1].expressions[0].op=='fptrunc_rz'
    assert sym[-1].expressions[0]!=Expr('cast',(Expr('input',(0,(0,)),'f32'),),'f16')


@pytest.mark.parametrize('call', [
    'T.call_extern("float32", "puts", x[0])',
    'T.call_extern("float32", tag, x[0], x[0])',
    'T.call_extern("float64", "__fdividef", x[0], x[0])',
    'T.call_extern("float32", "__fdividef", x[0])',
    'T.call_extern("float32", "__fdividef", x[0], T.float16(1))',
    'T.call_extern("float16", "__float2half_rz", T.int32(1))',
    'T.call_intrin("float32", "tir.call_extern", "__umulhi", x[0], x[0])',
    'T.call_pure_extern("float32", "__fdividef", x[0], x[0])',
])
def test_invalid_extern_rejected_during_preflight_in_dead_branch(call):
    source=tir([('x',(1,),'float32')],'tag = "__fdividef"\nif False:\n    '+call)
    with pytest.raises(Unsupported,match='tir.call_extern'):
        TIRInterpreter(source)
    assert any(i['details'].get('operation')=='tir.call_extern' for i in tir_inventory(source,'00_tir_initial')[1])


def test_extern_runtime_rejects_wrong_bound_dtype():
    args=[('x',(1,),'float32'),('y',(1,),'float16')]
    source=tir(args,'y[0] = T.call_extern("float16", "__float2half_rz", x[0])')
    with pytest.raises(Unsupported,match='dtype'):
        execute(TIRInterpreter(source),np.ones(1,np.float64),np.empty(1,np.float16))


def test_real_tvm_objects():
    tvm=pytest.importorskip('tilelang',reason='requires matching TileLang/TVM Python').tvm
    from tir_mlir_checker.tvm_interpreter import TVMInterpreter
    t=tvm.tir
    a=t.decl_buffer((4,),'float32',name='a')
    y=t.decl_buffer((4,),'float16',name='y')
    i=t.Var('i','int32')
    imm=lambda v:t.IntImm('int32',v)
    def call(dtype,name,*values):
        return t.Call(dtype,tvm.ir.Op.get(name),values)
    def load_region(b):
        return t.BufferLoad(b,[t.Ramp(imm(0),imm(1),imm(4))])
    scan=t.Evaluate(call('handle','tl.tileop.cumsum',load_region(a),load_region(y),imm(-1),t.IntImm('bool',False)))
    cvt=call('float16','tir.call_extern',t.StringImm('__float2half_rz'),t.BufferLoad(a,[i]))
    marker=t.Evaluate(call('handle','tl.wait_wgmma',imm(0)))
    loop=t.For(i,imm(0),imm(4),t.ForKind.SERIAL,t.SeqStmt([
        t.IfThenElse(t.EQ(i,imm(1)),t.Evaluate(call('handle','tl.loop_break')),None),t.BufferStore(y,cvt,[i])]))
    mapping={a.data:a,y.data:y}
    engine=TVMInterpreter(t.PrimFunc(list(mapping),t.SeqStmt([scan,marker,loop]),buffer_map=mapping))
    data=np.array([1.0+3*2.0**-11,2,3,4],np.float32)
    out,report=execute(engine,data,np.empty(4,np.float16))
    assert out[-1].values[0].view('uint16')==0x3c01
    np.testing.assert_array_equal(out[-1].writes,[2,1,1,1])
    assert report['reductions'][0]['kind']=='cumsum' and engine.loop_specs=={} and engine.loop_stack==[]
    assert any(v['operation']=='tl.wait_wgmma' for v in report['node_events'] if 'operation' in v)
    invalid=t.IfThenElse(t.IntImm('bool',False),t.Evaluate(call('float32','tir.call_extern',t.StringImm('unknown'))),None)
    with pytest.raises(Unsupported,match='tir.call_extern'):
        TVMInterpreter(t.PrimFunc(list(mapping),invalid,buffer_map=mapping))

"""Phase 2 scalar, tile reduction, and bounded block metadata contracts."""
import importlib

import numpy as np
import pytest

from tests.test_batch2_tile_ops import execute, mlir, tir
from tir_mlir_checker.execution import (
    SemanticFault, Unsupported, expression_digest, literal, mlir_cast, scalar_op,
)
from tir_mlir_checker.mlir_interpreter import MLIRInterpreter
from tir_mlir_checker.rules import tir_inventory
from tir_mlir_checker.tir_interpreter import TIRInterpreter


@pytest.mark.parametrize('op,value,target,bits', [
    ('fptosi', -128.99, 'int8', 128), ('fptosi', 127.99, 'int8', 127),
    ('fptoui', -0.99, 'int8', 0), ('fptoui', 255.99, 'int8', 255),
    ('fptosi', -2.0**63, 'int64', 2**63),
    ('fptoui', np.nextafter(2.0**64, 0.), 'int64', 2**64-2048),
])
def test_float_to_integer_truncates_before_range_check(op, value, target, bits):
    result = mlir_cast(op, value, 'f64', target)
    assert int(result.view(f'uint{8*result.dtype.itemsize}')) == bits


@pytest.mark.parametrize('op,value,target', [
    ('fptosi', -129., 'i8'), ('fptosi', 128., 'i8'), ('fptosi', 2.0**63, 'i64'),
    ('fptoui', -1., 'i8'), ('fptoui', 256., 'i8'), ('fptoui', 2.0**64, 'i64'),
    *((op, value, 'i32') for op in ('fptosi', 'fptoui') for value in (np.nan, np.inf, -np.inf)),
])
def test_invalid_float_to_int_is_a_semantic_fault(op, value, target):
    with pytest.raises(SemanticFault) as exc:
        mlir_cast(op, value, 'f64', target)
    assert exc.value.rule_id == 'ARITH-INVALID-CAST'


def cast_program(op):
    return mlir([('x','memref<1xf32>'),('y','memref<1xi8>')], f'''    %i = arith.constant 0 : index
    %v = memref.load %x[%i] : memref<1xf32>
    %r = arith.{op} %v : f32 to i8
    memref.store %r, %y[%i] : memref<1xi8>''')


def check_cast_frontend(engine):
    out = np.zeros(1,np.int8)
    fingerprints = []
    for name,value,expected in [('fptosi',-3.75,-3), ('fptoui',253.75,-3)]:
        program = cast_program(name)
        x = np.array([value],np.float32)
        result,_ = execute(engine(program),x,out)
        assert result[1].values[0] == expected
        result,_ = execute(engine(program),x,out,symbolic=True)
        assert result[1].expressions[0].op == name
        fingerprints.append(expression_digest(result[1].expressions))
        with pytest.raises(SemanticFault) as exc:
            execute(engine(program),np.array([np.nan],np.float32),out)
        assert exc.value.rule_id == 'ARITH-INVALID-CAST'
    assert fingerprints[0] != fingerprints[1]


def clz_program():
    return mlir([('x','memref<1xi32>'),('y','memref<1xi32>')], '''    %i = arith.constant 0 : index
    %v = memref.load %x[%i] : memref<1xi32>
    %r = math.ctlz %v : i32
    memref.store %r, %y[%i] : memref<1xi32>''')


@pytest.mark.parametrize('kind,value,expected', [
    ('int8',0,8), ('int8',-1,0), ('uint8',64,1), ('int32',0,32),
    ('int32',-2147483648,0), ('uint32',1,31), ('int64',0,64), ('int64',2**61,2),
])
def test_clz_uses_operand_width_and_unsigned_bits(kind,value,expected):
    v = np.asarray(value,dtype=kind)[()]
    assert scalar_op('ctlz',[v],'int32',[kind]) == expected


def test_text_executes_cast_and_clz_with_symbolic_agreement():
    check_cast_frontend(MLIRInterpreter)
    x,y = np.zeros(1,np.int32),np.zeros(1,np.int32)
    models = [MLIRInterpreter(clz_program()),
              TIRInterpreter(tir([('x',(1,),'int32'),('y',(1,),'int32')],'y[0] = T.clz(x[0])'))]
    fingerprints=[]
    for engine in models:
        result,_ = execute(engine,x,y)
        assert result[1].values[0] == 32
        result,_ = execute(engine,x,y,symbolic=True)
        fingerprints.append(expression_digest(result[1].expressions))
    assert fingerprints[0] == fingerprints[1]
    value = literal(np.float32(4))
    assert mlir_cast('fptosi',value,'f32','i32') != mlir_cast('fptoui',value,'f32','i32')


REDUCE_EXAMPLES = [
    ('sum',(-3,5,-6),-4,2), ('abssum',(-3,5,-6),14,20),
    ('max',(-3,5,-6),5,6), ('absmax',(-3,5,-7),7,7),
    ('min',(3,5,7),3,3), ('bitand',(14,10,11),10,2),
    ('bitor',(1,8,16),25,31), ('bitxor',(3,12,5),10,12),
]


def reduce_text(kind,clear,keep=True,target='int32',source='int32',dim=-1):
    return tir([('x',(1,3),source),('y',(1,1) if keep else (1,),target)],
               f'T.reduce(x, y, "{kind}", {dim}, T.bool({clear!r}))')


@pytest.mark.parametrize('kind,values,cleared,accumulated', REDUCE_EXAMPLES)
def test_tile_reduce_all_kinds_all_initializers(kind,values,cleared,accumulated):
    x = np.array([values],np.int32)
    for clear,expected in ((True,cleared),(False,accumulated)):
        keep = not clear
        y = np.full((1,1) if keep else (1,),6,np.int32)
        source = reduce_text(kind,clear,keep=keep)
        assert not tir_inventory(source,'02_tir_optimized')[1]
        engine = TIRInterpreter(source)
        result,info = execute(engine,x,y,inout=True)
        assert result[1].values.item() == expected
        details = info['reductions'][0]
        assert (details['axis'],details['clear'],details['output_keeps_reduced_dim']) == (1,clear,keep)
        result,_ = execute(engine,x,y,inout=True,symbolic=True)
        assert len(expression_digest(result[1].expressions)) == 64
        if not clear:
            with pytest.raises(SemanticFault) as exc:
                execute(engine,x,y)
            assert exc.value.rule_id == 'MEM-UNINITIALIZED'


@pytest.mark.parametrize('kind', ('max','min','absmax'))
def test_ordered_reduction_nan_and_zero(kind):
    program = TIRInterpreter(reduce_text(kind,True,target='float32',source='float32'))
    out = np.zeros((1,1),np.float32)
    for raw,expected in [([np.nan,np.nan,-0.0],0x80000000),([np.nan,-0.0,0.0],0),
                         ([0., 0., np.nan],0x7FC00001)]:
        x = np.array([raw],np.float32)
        if np.isnan(raw[-1]):
            x.view(np.uint32)[0,-1] = 0x7FC00001
        result,_ = execute(program,x,out)
        assert result[1].values.view(np.uint32).item() == expected


def test_reduction_checks_axis_shape_and_operand_types():
    x,y = np.ones((1,3),np.float16),np.zeros((1,1),np.float32)
    engine = TIRInterpreter(reduce_text('abssum',True,target='float32',source='float16'))
    result,_ = execute(engine,x,y)
    assert result[1].values.item() == 3
    for kind,axis in [('bitxor',-1), ('max',-3), ('unknown',-1)]:
        with pytest.raises(Unsupported):
            execute(TIRInterpreter(reduce_text(kind,True,target='float32',source='float16',dim=axis)),x,y)


def test_real_mlir_polish_objects():
    ir = pytest.importorskip('mlir.ir')
    from tir_mlir_checker.mlir_frontend import DIALECTS
    from tir_mlir_checker.mlir_object_interpreter import MLIRObjectInterpreter
    for dialect in DIALECTS:
        importlib.import_module('mlir.dialects.'+dialect)

    def engine(source):
        ctx = ir.Context()
        with ctx:
            ctx.load_all_available_dialects()
            module = ir.Module.parse(source)
            assert module.operation.verify()
        return MLIRObjectInterpreter((ctx,module),source)

    check_cast_frontend(engine)
    result,_ = execute(engine(clz_program()),np.array([-1],np.int32),np.zeros(1,np.int32))
    assert result[1].values[0] == 0


def test_real_tvm_polish_objects():
    t = pytest.importorskip('tilelang').tvm
    from tilelang.layout import Layout
    from tir_mlir_checker.tvm_interpreter import TVMInterpreter
    T = t.tir
    a = T.decl_buffer((1,3),'int32',name='a')
    y = T.decl_buffer((1,1),'int32',name='y')
    mapping = {a.data:a,y.data:y}
    args = [T.BufferLoad(a,[0,T.Ramp(0,1,3)]), T.BufferLoad(y,[0,0]),
            T.StringImm('bitxor'), T.IntImm('int32',-1), T.IntImm('bool',False)]
    node = T.Evaluate(T.Call('handle',t.ir.Op.get('tl.tileop.reduce'),args))
    engine = TVMInterpreter(T.PrimFunc(list(mapping),node,buffer_map=mapping))
    x,out = np.array([[3,12,5]],np.int32),np.array([[6]],np.int32)
    result,_ = execute(engine,x,out,inout=True)
    assert result[1].values.item() == 12
    with pytest.raises(SemanticFault,match='未初始化'):
        execute(engine,x,out)

    local = T.decl_buffer((2,2),'int32',name='local_mat',scope='shared.dyn')
    clz = T.Call('int32',t.ir.Op.get('tir.clz'),[T.BufferLoad(a,[0,0])])
    body = T.SeqStmt([T.BufferStore(local,clz,[1,1]),T.BufferStore(y,T.BufferLoad(local,[1,1]),[0,0])])

    def function(annotations, matches=None):
        block = T.Block([],[],[],'local_tile',body,alloc_buffers=[local],annotations=annotations,
                        match_buffers=matches)
        return T.PrimFunc(list(mapping),block,buffer_map=mapping)

    layout = Layout([2,2], lambda i,j: i*4+j)
    fun = function({'layout_map':{local.data:layout}})
    engine = TVMInterpreter(t.ir.load_json(t.ir.save_json(fun)))
    x[0,0]=0
    result,_ = execute(engine,x,out)
    assert result[1].values.item() == 32
    result,_ = execute(engine,x,out,symbolic=True)
    assert result[1].expressions.item().op == 'ctlz'

    for annotations in ({'unreviewed':T.IntImm('int32',1)}, {'layout_map':{a.data:layout}},
                        {'layout_map':{local.data:Layout([2,2], lambda i,j: i+j)}}):
        with pytest.raises(Unsupported,match='注解|layout_map'):
            TVMInterpreter(function(annotations))
    match = T.MatchBufferRegion(T.decl_buffer((1,3),'int32'),T.BufferRegion(a,[t.ir.Range(0,1),t.ir.Range(0,3)]))
    with pytest.raises(Unsupported,match='match_buffers'):
        TVMInterpreter(function({},[match]))

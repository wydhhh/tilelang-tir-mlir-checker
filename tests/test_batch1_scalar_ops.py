"""Scalar contracts and frontend wiring. Real object tests have optional dependencies."""
import importlib

import numpy as np
import pytest

from tir_mlir_checker.execution import (
    Budget, Expr, SemanticFault, Unsupported, cast, expression_digest, make_buffers,
    mlir_cast, scalar_op, tir_scalar,
)
from tir_mlir_checker.mlir_interpreter import MLIRInterpreter
from tir_mlir_checker.tir_interpreter import TIRInterpreter


# Math witnesses are exact or independently evaluated at the expected precision.
MATH_CASES = [
    ('log2', 'log2', (8.0,), 3.0),
    ('erf', 'erf', (-1.0,), -0.8427007929497149),
    ('floor', 'floor', (-2.25,), -3.0),
    ('sin', 'sin', (0.5,), 0.479425538604203),
    ('cos', 'cos', (0.5,), 0.877582561890373),
    ('nearbyint', 'roundeven', (-2.5,), -2.0),
    ('trunc', 'trunc', (-2.75,), -2.0),
    ('log1p', 'log1p', (1.0,), 0.6931471805599453),
    ('ceil', 'ceil', (-2.75,), -2.0),
    ('copysign', 'copysign', (2.5, -0.0), -2.5),
    ('pow', 'powf', (9.0, 0.5), 3.0),
    ('fmod', 'arith.remf', (-5.5, 2.0), -1.5),
    ('round', None, (-2.5,), -3.0),
    ('ieee_fdiv', 'arith.divf', (7.0, -2.0), -3.5),
    ('ieee_fmaf', 'fma', (1 + 2.0**-12, 1 + 2.0**-12, 2.0**-54), 1 + 2.0**-11 + 2.0**-23),
]
TIR_INT_CASES = [
    ('bitwise_and', 'int32', (-16, 0x35), 0x30),
    ('bitwise_or', 'int32', (-16, 0x05), -11),
    ('bitwise_xor', 'uint32', (0xFFFF_0000, 0xC000_0001), 0x3FFF_0001),
    ('bitwise_not', 'uint32', (0x0FFF_FFFF,), 0xF000_0000),
    ('shift_left', 'int32', (0x4000_0001, 1), -2147483646),
    ('shift_right', 'int32', (-3, 1), -2),
    ('shift_right', 'uint32', (0xFFFF_FFFD, 1), 0x7FFF_FFFE),
    ('popcount', 'int64', (-2,), 63),
    ('pow', 'int32', (2, 32), 0),
]
MLIR_INT_CASES = [
    ('arith.andi', 'i1', (True, False), False),
    ('arith.ori', 'i1', (False, True), True),
    ('arith.andi', 'i32', (-16, 0x35), 0x30),
    ('arith.ori', 'i32', (-16, 5), -11),
    ('arith.xori', 'i32', (-1, 0x7FFF_FFFF), -2147483648),
    ('arith.shli', 'i32', (0x4000_0001, 1), -2147483646),
    ('arith.shrsi', 'i32', (-3, 1), -2),
    ('arith.shrui', 'i32', (-3, 1), 0x7FFF_FFFE),
    ('arith.divsi', 'i64', (-9223372036854775807, 3), -3074457345618258602),
    ('arith.divui', 'i64', (-1, 3), 6148914691236517205),
    ('arith.remui', 'i64', (-2, 3), 2),
    ('arith.minsi', 'i32', (-1, 2), -1),
    ('arith.maxsi', 'i32', (-1, 2), 2),
    ('arith.minui', 'i32', (-1, 2), 2),
    ('arith.maxui', 'i32', (-1, 2), -1),
    ('math.ctpop', 'i64', (-2,), 63),
]
MLIR_DTYPES = {'i1': 'bool', 'i16': 'int16', 'i32': 'int32', 'i64': 'int64',
               'f16': 'float16', 'f32': 'float32', 'f64': 'float64'}


def buffers(values, kinds, result, symbolic=False):
    args = tuple(np.array([x], dtype=k) for x, k in zip(values, kinds))
    args += (np.empty(1, dtype=result),)
    return make_buffers(args, (len(values),), symbolic, Budget())


def run(engine, values, kinds, result, symbolic=False):
    storage = buffers(values, kinds, result, symbolic)
    engine.run(storage, symbolic=symbolic)
    assert storage[-1].writes[0] == 1
    return storage[-1].expressions[0] if symbolic else storage[-1].values[0]


def tvmscript(expression, kinds, result):
    types = [f'a{i}: T.Buffer((1,), "{k}")' for i, k in enumerate(kinds)]
    types.append(f'y: T.Buffer((1,), "{result}")')
    return ('@I.ir_module\nclass Module:\n    @T.prim_func\n'
            f'    def f({", ".join(types)}):\n        y[0] = {expression}\n')


def tir_call(name, arity):
    args = ', '.join(f'a{i}[0]' for i in range(arity))
    return f'T.{name}({args})'


def mlir_source(op, kinds, result):
    params = [f'%a{i}: memref<1x{k}>' for i, k in enumerate(kinds)]
    params.append(f'%y: memref<1x{result}>')
    lines = ['module {', f'  func.func @f({", ".join(params)}) {{', '    %i = arith.constant 0 : index']
    lines.extend(f'    %v{i} = memref.load %a{i}[%i] : memref<1x{k}>' for i, k in enumerate(kinds))
    args = ', '.join(f'%v{i}' for i in range(len(kinds)))
    annotation = f'{kinds[0]} to {result}' if op == 'arith.bitcast' else result
    lines += [f'    %r = {op} {args} : {annotation}', f'    memref.store %r, %y[%i] : memref<1x{result}>',
              '    return', '  }', '}']
    return '\n'.join(lines)


def mlir_name(name):
    return name if '.' in name else 'math.' + name


def execute_mlir(factory, op, kind, values, result=None, symbolic=False):
    result = result or kind
    kinds = [kind] * len(values)
    return run(factory(mlir_source(op, kinds, result)), values,
               [MLIR_DTYPES[k] for k in kinds], MLIR_DTYPES[result], symbolic=symbolic)


@pytest.mark.parametrize('x,is_nan,is_finite', [
    (np.nan, True, False), (np.inf, False, False), (-np.inf, False, False),
    (-0.0, False, True), (1.5, False, True),
])
def test_riscv_float_predicate_lowering(x, is_nan, is_finite):
    footer = """
    memref.store %r, %y[%i] : memref<1xi1>
    return
  }
}
"""
    header = """module {
  func.func @f(%x: memref<1xf32>, %y: memref<1xi1>) {
    %i = arith.constant 0 : index
    %v = memref.load %x[%i] : memref<1xf32>
"""
    recipes = [
        ('isnan', '    %r = arith.cmpf uno, %v, %v : f32\n', is_nan),
        ('isfinite', """    %a = math.absf %v : f32
    %inf = arith.constant 0x7F800000 : f32
    %r = arith.cmpf one, %a, %inf : f32
""", is_finite),
    ]
    for op, body, expected in recipes:
        tir = TIRInterpreter(tvmscript(tir_call(op, 1), ['float32'], 'bool'))
        mlir = MLIRInterpreter(header + body + footer)
        assert bool(run(tir, (x,), ['float32'], 'bool')) is expected
        assert bool(run(mlir, (x,), ['float32'], 'bool')) is expected


@pytest.mark.parametrize('tir_name,mlir_op,values,expected', MATH_CASES)
def test_float_text_frontends(tir_name, mlir_op, values, expected):
    kinds = ['float32'] * len(values)
    tir = TIRInterpreter(tvmscript(tir_call(tir_name, len(values)), kinds, 'float32'))
    actual = run(tir, values, kinds, 'float32')
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=0)
    if mlir_op:
        other = execute_mlir(MLIRInterpreter, mlir_name(mlir_op), 'f32', values)
        np.testing.assert_array_equal(actual, other)


@pytest.mark.parametrize('name,kind,values,expected', TIR_INT_CASES)
def test_tir_integer_calls(name, kind, values, expected):
    kinds = [kind] * len(values)
    engine = TIRInterpreter(tvmscript(tir_call(name, len(values)), kinds, kind))
    assert run(engine, values, kinds, kind) == expected


@pytest.mark.parametrize('op,kind,values,expected', MLIR_INT_CASES)
def test_mlir_integer_ops(op, kind, values, expected):
    assert execute_mlir(MLIRInterpreter, op, kind, values) == expected


@pytest.mark.parametrize('op,kind,a,b,rule', [
    ('arith.divsi', 'i32', -2147483648, -1, 'ARITH-INVALID-DIVISION'),
    ('arith.divsi', 'i64', -(2**63), -1, 'ARITH-INVALID-DIVISION'),
    ('arith.divsi', 'i64', 1, 0, 'ARITH-INVALID-DIVISION'),
    ('arith.divui', 'i64', -1, 0, 'ARITH-INVALID-DIVISION'),
    ('arith.remui', 'i32', -1, 0, 'ARITH-INVALID-DIVISION'),
    ('arith.shli', 'i32', 1, -1, 'ARITH-INVALID-SHIFT'),
    ('arith.shrsi', 'i64', -1, 64, 'ARITH-INVALID-SHIFT'),
    ('arith.shrui', 'i32', -1, 32, 'ARITH-INVALID-SHIFT'),
])
def test_mlir_integer_faults(op, kind, a, b, rule):
    with pytest.raises(SemanticFault) as error:
        execute_mlir(MLIRInterpreter, op, kind, (a, b))
    assert error.value.rule_id == rule
    assert error.value.line == 6


@pytest.mark.parametrize('operator,amount', [('<<', -1), ('<<', 32), ('>>', -1), ('>>', 32)])
def test_tir_invalid_shifts(operator, amount):
    kinds = ['int32', 'int32']
    engine = TIRInterpreter(tvmscript(f'a0[0] {operator} a1[0]', kinds, 'int32'))
    with pytest.raises(SemanticFault) as error:
        run(engine, (-1, amount), kinds, 'int32')
    assert error.value.rule_id == 'ARITH-INVALID-SHIFT'
    assert error.value.details == {'shift': amount, 'bitwidth': 32}
    # The 8-bit TIR contract rejects a count of 8 even if the target lowerer
    # uses a wider intermediate operation.
    with pytest.raises(SemanticFault, match='位宽'):
        tir_scalar('tir.shift_left', [1, 8], 'int8', ['int8', 'int32'])


@pytest.mark.parametrize('kind', ['float16', 'float32', 'float64'])
def test_rounding_family_retains_signed_zero(kind):
    x = cast(-0.5, kind)
    assert tir_scalar('tir.round', [x], kind) == -1
    nearest = tir_scalar('tir.nearbyint', [x], kind)
    assert nearest == 0 and np.signbit(nearest)
    assert scalar_op('roundeven', [cast(2.5, kind)], kind) == 2
    below = np.nextafter(cast(0.5, kind), cast(0.0, kind))
    assert tir_scalar('tir.round', [below], kind) == 0
    for special in (np.inf, -np.inf):
        assert tir_scalar('tir.round', [cast(special, kind)], kind) == special
    assert np.isnan(tir_scalar('tir.round', [cast(np.nan, kind)], kind))


@pytest.mark.parametrize('x,expected', [(np.nan, (True, False)), (np.inf, (False, False)),
                                       (-np.inf, (False, False)), (-0.0, (False, True)),
                                       (2.0, (False, True))])
def test_predicates_and_infinity(x, expected):
    for name, truth in zip(('isnan', 'isfinite'), expected):
        model = TIRInterpreter(tvmscript(f'T.{name}(a0[0])', ['float16'], 'bool'))
        y = run(model, (x,), ['float16'], 'bool')
        assert isinstance(y, np.bool_) and bool(y) is truth
    assert np.isposinf(tir_scalar('tl.infinity', ['float16'], 'float16'))
    assert np.isposinf(tir_scalar('tir.infinity', [], 'float64'))
    # A typed Call can carry a hint; its declared result determines the dtype.
    assert tir_scalar('tl.infinity', ['float32'], 'float16').dtype == np.dtype('float16')
    assert np.isposinf(run(TIRInterpreter(tvmscript(
        'T.call_intrin("float32", "tir.infinity")', [], 'float32')), (), [], 'float32'))
    with pytest.raises(Unsupported):
        tir_scalar('tl.infinity', ['int32'], 'int32')
    with pytest.raises(Unsupported):
        tir_scalar('tl.infinity', [1], 'float32')


def test_fmod_sign_and_ieee_division():
    cases = [(5.5, -2.0, 1.5), (-5.5, 2.0, -1.5), (-4.0, 2.0, -0.0)]
    for a, b, expected in cases:
        y = scalar_op('fmod', [np.float16(a), np.float16(b)], 'f16')
        assert y == expected and np.signbit(y) == np.signbit(expected)
    assert np.isnan(scalar_op('fmod', [np.float32(1), np.float32(0)], 'f32'))
    assert np.isneginf(tir_scalar('tl.ieee_fdiv', [np.float32(1), np.float32(-0.0), 'rn'], 'f32'))
    with pytest.raises(Unsupported, match='rn'):
        tir_scalar('tl.ieee_fdiv', [np.float32(1), np.float32(2), 'rz'], 'f32')
    with pytest.raises(Unsupported, match='rn'):
        tir_scalar('tl.ieee_fmaf', [np.float32(1)] * 3, 'f32')


@pytest.mark.parametrize('kind,precision', [('f16', 10), ('f32', 23), ('f64', 52), ('bf16', 7)])
def test_fma_is_fused_at_destination_precision(kind, precision):
    if kind == 'bf16':
        pytest.importorskip('ml_dtypes')
    delta = 2.0**-precision
    a, b, c = [cast(v, kind) for v in (1 + delta, 1 - delta, -1.0)]
    result = scalar_op('fma', [a, b, c], kind)
    assert result == cast(-delta**2, kind)
    assert result != cast(cast(a * b, kind) + c, kind)
    # A finite product that would otherwise overflow can cancel back to finite.
    maximum = cast((2.0 - delta) * 2.0**({'f16': 15, 'f32': 127, 'bf16': 127, 'f64': 1023}[kind]), kind)
    assert scalar_op('fma', [maximum, cast(2, kind), -maximum], kind) == maximum
    assert np.isposinf(scalar_op('fma', [maximum, cast(2, kind), cast(0, kind)], kind))


def test_fma_special_values_and_double_rounding():
    def fma(a, b, c):
        return scalar_op('fma', list(map(np.float32, (a, b, c))), 'f32')
    y = fma(1 + 2.0**-12, 1 + 2.0**-12, 2.0**-54)
    assert y.view(np.uint32) == 0x3F801001  # just above a binary32 midpoint
    assert np.isnan(fma(0, np.inf, 1))
    assert np.isnan(fma(np.inf, 1, -np.inf))
    assert np.isnan(fma(np.nan, 1, 2))
    assert np.isneginf(fma(np.inf, -1, 2))
    assert np.signbit(fma(-0.0, 1, -0.0))
    assert not np.signbit(fma(-0.0, 1, 0.0))
    assert not np.signbit(fma(1, -1, 1))
    assert fma(2.0**-149, 0.5, 0).view(np.uint32) == 0
    assert fma(-2.0**-149, 0.5, 0).view(np.uint32) == 0x80000000
    assert fma(2.0**-149, 0.75, 0).view(np.uint32) == 1


@pytest.mark.parametrize('unsigned,float_kind,payload', [
    ('uint16', 'f16', 0xFE15), ('uint16', 'bf16', 0x8000),
    ('uint32', 'f32', 0x7FC12345), ('uint64', 'f64', 0x8000000000000000),
    ('uint8', 'f8E4M3FN', 0xFF),
])
def test_bitcast_preserves_all_bits(unsigned, float_kind, payload):
    if float_kind in {'f8E4M3FN', 'bf16'}:
        pytest.importorskip('ml_dtypes')
    value = cast(payload, unsigned)
    floated = tir_scalar('tir.reinterpret', [value], float_kind, [unsigned])
    assert floated.tobytes() == value.tobytes()
    back = mlir_cast('bitcast', floated, float_kind, unsigned)
    assert back == payload
    with pytest.raises(Unsupported, match='位宽'):
        mlir_cast('bitcast', floated, float_kind, 'i1')


def test_bitcast_text_frontends_and_sign_transfer():
    payload = 0x7FC12345
    y = execute_mlir(MLIRInterpreter, 'arith.bitcast', 'i32', (payload,), 'f32')
    assert y.view(np.uint32) == payload
    tir = TIRInterpreter(tvmscript('T.reinterpret(a0[0], "float32")', ['uint32'], 'float32'))
    assert run(tir, (payload,), ['uint32'], 'float32').view(np.uint32) == payload
    assert scalar_op('copysign', [y, np.float32(-0.0)], 'f32').view(np.uint32) == (payload | 0x80000000)
    with pytest.raises(Unsupported, match='位宽'):
        execute_mlir(MLIRInterpreter, 'arith.bitcast', 'i32', (payload,), 'f16')


def test_shared_symbolic_nodes_distinguish_rounding_and_signedness():
    def symbolic_tir(expr, kinds, result, values):
        return run(TIRInterpreter(tvmscript(expr, kinds, result)), values, kinds, result, symbolic=True)
    for kind, op in [('int32', 'arith.shrsi'), ('uint32', 'arith.shrui')]:
        tir = symbolic_tir('a0[0] >> a1[0]', [kind, kind], kind, (10, 1))
        mlir = execute_mlir(MLIRInterpreter, op, 'i32', (10, 1), symbolic=True)
        assert tir == mlir and tir.fingerprint() == mlir.fingerprint()
    signed = execute_mlir(MLIRInterpreter, 'arith.shrsi', 'i32', (-3, 1), symbolic=True)
    unsigned = execute_mlir(MLIRInterpreter, 'arith.shrui', 'i32', (-3, 1), symbolic=True)
    assert signed != unsigned and signed.fingerprint() != unsigned.fingerprint()
    near = symbolic_tir('T.nearbyint(a0[0])', ['float32'], 'float32', (2.5,))
    away = symbolic_tir('T.round(a0[0])', ['float32'], 'float32', (2.5,))
    even = execute_mlir(MLIRInterpreter, 'math.roundeven', 'f32', (2.5,), symbolic=True)
    assert even == near and even != away
    a = Expr('input', (0, (0,)), 'i32')
    n = scalar_op('ctpop', [a], 'i32')
    f = mlir_cast('bitcast', n, 'i32', 'f32')
    expr = scalar_op('fma', [f, np.float32(2.0), np.float32(1.0)], 'f32')
    assert expr.data()[0:2] == ['fma', 'f32']
    assert len(expression_digest(np.array([expr], dtype=object))) == 64
    assert tir_scalar('tir.isfinite', [f], 'bool').type == 'i1'
    with pytest.raises(SemanticFault, match='位宽'):
        scalar_op('shl', [a, 32], 'i32', ['i32', 'i32'])


def test_unknown_intrinsic_is_still_rejected_during_preflight():
    script = tvmscript('T.call_intrin("int32", "tir.unimplemented_example", a0[0])', ['int32'], 'int32')
    script = script.replace('        y[0]', '        if False:\n            y[0]')
    with pytest.raises(Unsupported, match='字面量标量操作'):
        TIRInterpreter(script)
    from tir_mlir_checker.rules import tir_inventory
    assert any(r['rule_id'] == 'CAP-TIR-CALL' for r in tir_inventory(script, '00_tir_initial')[1])
    source = mlir_source('math.fma', ['f32'] * 2, 'f32')
    with pytest.raises(Unsupported, match='语法'):
        MLIRInterpreter(source)


def test_tvm_real_objects():
    tilelang = pytest.importorskip('tilelang', reason='requires matching TileLang/TVM Python')
    from tir_mlir_checker.tvm_interpreter import TVMInterpreter
    tvm = tilelang.tvm
    def engine(op, kinds, result, controls=()):
        tir = tvm.tir
        src = [tir.decl_buffer((1,), k, name=f'a{i}') for i, k in enumerate(kinds)]
        dst = tir.decl_buffer((1,), result, name='y')
        args = [tir.BufferLoad(b, [0]) for b in src] + [tir.StringImm(s) for s in controls]
        call = tir.Call(result, tvm.ir.Op.get(op), args)
        mapping = {b.data: b for b in [*src, dst]}
        f = tir.PrimFunc(list(mapping), tir.BufferStore(dst, call, [0]), buffer_map=mapping)
        assert tvm.tir.analysis.verify_ssa(f) and tvm.tir.analysis.verify_well_formed(f)
        return TVMInterpreter(f)
    for name, _, vals, expected in MATH_CASES:
        controls = ('rn',) if name.startswith('ieee_') else ()
        prefix = 'tl.' if controls else 'tir.'
        ks = ['float32'] * len(vals)
        actual = run(engine(prefix + name, ks, 'float32', controls), vals, ks, 'float32')
        np.testing.assert_allclose(actual, expected, rtol=2e-7)
    for name, kind, vals, expected in TIR_INT_CASES:
        ks = [kind] * len(vals)
        assert run(engine('tir.' + name, ks, kind), vals, ks, kind) == expected
    assert np.isposinf(run(engine('tl.infinity', [], 'float32', ['float32']), (), [], 'float32'))
    assert np.isposinf(run(engine('tl.infinity', [], 'float32'), (), [], 'float32'))
    assert run(engine('tir.isnan', ['float32'], 'bool'), (np.nan,), ['float32'], 'bool')
    assert not run(engine('tir.isfinite', ['float32'], 'bool'), (np.inf,), ['float32'], 'bool')
    y = run(engine('tir.reinterpret', ['uint32'], 'float32'), (0x7FC12345,), ['uint32'], 'float32')
    assert y.view(np.uint32) == 0x7FC12345
    with pytest.raises(SemanticFault) as error:
        run(engine('tir.shift_right', ['int32', 'int32'], 'int32'), (-1, 32), ['int32'] * 2, 'int32')
    assert error.value.rule_id == 'ARITH-INVALID-SHIFT' and error.value.details.get('tvm_location')


def test_mlir_real_objects():
    ir = pytest.importorskip('mlir.ir', reason='requires official MLIR Python bindings')
    from tir_mlir_checker.mlir_frontend import DIALECTS
    from tir_mlir_checker.mlir_object_interpreter import MLIRObjectInterpreter
    for dialect in DIALECTS:
        importlib.import_module('mlir.dialects.' + dialect)
    def engine(source):
        ctx = ir.Context()
        with ctx:
            ctx.allow_unregistered_dialects = False
            ctx.load_all_available_dialects()
            module = ir.Module.parse(source)
            assert module.operation.verify()
        return MLIRObjectInterpreter((ctx, module), source)
    for _, op, vals, expected in MATH_CASES:
        if op:
            actual = execute_mlir(engine, mlir_name(op), 'f32', vals)
            np.testing.assert_allclose(actual, expected, rtol=2e-7)
    for op, kind, vals, expected in MLIR_INT_CASES:
        assert execute_mlir(engine, op, kind, vals) == expected
    y = execute_mlir(engine, 'arith.bitcast', 'i32', (0x7FC12345,), 'f32')
    assert y.view(np.uint32) == 0x7FC12345
    with pytest.raises(SemanticFault) as error:
        execute_mlir(engine, 'arith.divsi', 'i64', (-(2**63), -1))
    assert error.value.rule_id == 'ARITH-INVALID-DIVISION'
    assert error.value.details.get('mlir_location')
    values = (2.5,)
    assert (execute_mlir(engine, 'math.roundeven', 'f32', values, symbolic=True)
            == execute_mlir(MLIRInterpreter, 'math.roundeven', 'f32', values, symbolic=True))

"""Public low-precision, stride and scalar contracts with independent references."""
import unittest
import numpy as np
from tir_mlir_checker.execution import dtype, storage_type, Budget, make_buffers
from tir_mlir_checker.cases import TestCase, Comparison, validate_case, compare
from tir_mlir_checker.tir_interpreter import TIRInterpreter
from tir_mlir_checker.mlir_interpreter import MLIRInterpreter


class ScalarContractTests(unittest.TestCase):
    def test_fp8_all_encodings_extend_to_expected_ieee_values(self):
        # Independent E4M3FN decoder: subnormal exp=0, bias=7; exp=15
        # remains finite except the all-one mantissa NaN encoding.
        d=dtype('f8E4M3FN');self.assertEqual(storage_type(d),'f8E4M3FN')
        bits=np.arange(256,dtype='uint8');x=bits.view(d)
        expected=[]
        for b in range(256):
            sign=-1 if b&128 else 1;e=(b>>3)&15;m=b&7
            expected.append(np.nan if e==15 and m==7 else sign*((m/8)*2**(-6) if e==0 else (1+m/8)*2**(e-7)))
        np.testing.assert_array_equal(x.astype('float32'),np.array(expected,dtype='float32'))
        self.assertTrue(np.signbit(x[128].astype('float32')))
        y=x.astype('float32')
        case=TestCase('all-bits',(x,np.empty_like(y)),(1,),(y,),comparison=Comparison('tolerance',equal_nan=True,signed_zero_equal=True))
        validate_case(case)
        self.assertTrue(compare(x,x,Comparison('tolerance',equal_nan=True,signed_zero_equal=True))['match'])

    def test_fp8_mlir_extf_executes_without_integer_alias(self):
        source='''module {
  func.func @f(%arg0: memref<4xf8E4M3FN>, %arg1: memref<4xf32>) {
    %z = arith.constant 0 : index
    %n = arith.constant 4 : index
    %one = arith.constant 1 : index
    scf.for %i = %z to %n step %one {
      %x = memref.load %arg0[%i] : memref<4xf8E4M3FN>
      %v = arith.extf %x : f8E4M3FN to f32
      memref.store %v, %arg1[%i] : memref<4xf32>
    }
    return
  }
}'''
        x=np.array([1,-2,0.125,448],dtype=dtype('f8E4M3FN'))
        b=make_buffers((x,np.empty(4,dtype='float32')),(1,),False,Budget())
        MLIRInterpreter(source).run(b,{},False)
        np.testing.assert_array_equal(b[1].values,[1,-2,0.125,448])

    def test_string_stride_symbol_is_bound_and_conflicts_rejected(self):
        source='''@I.ir_module
class Module:
    @T.prim_func
    def f(a: T.handle, b: T.handle):
        x=T.match_buffer(a,(2,1),strides=("s",1))
        y=T.match_buffer(b,(2,1))
        for i in T.serial(2):
            y[i,0]=x[i,0]
'''
        x=np.array([[2],[3]],dtype='float32')
        b=make_buffers((x,np.empty_like(x)),(1,),False,Budget())
        TIRInterpreter(source).run(b,{},False)
        np.testing.assert_array_equal(b[1].values,x)
        from tir_mlir_checker.execution import Unsupported
        with self.assertRaises(Unsupported):
            TIRInterpreter(source).run(b,{'s':2},False)

    def test_exp2_exact_powers_and_special_values(self):
        from tir_mlir_checker.execution import apply
        for exponent in (-149,-126,-3,0,1,20,127):
            actual=apply('exp2',[np.float32(exponent)],'f32')
            self.assertEqual(actual,np.float32(2.0**exponent))
            self.assertEqual(actual.dtype,np.dtype('float32'))
        self.assertEqual(apply('exp2',[np.float32(-np.inf)],'f32'),0)
        self.assertTrue(np.isposinf(apply('exp2',[np.float32(np.inf)],'f32')))
        self.assertTrue(np.isnan(apply('exp2',[np.float32(np.nan)],'f32')))

    def test_tir_infinity_and_exp2(self):
        source='''@I.ir_module
class Module:
    @T.prim_func
    def f(y:T.Buffer((3,),"float32")):
        y[0]=T.infinity("float32")
        y[1]=T.infinity("float32")*T.float32(-1)
        y[2]=T.exp2(T.float32(3))
'''
        b=make_buffers((np.empty(3,dtype='float32'),),(0,),False,Budget())
        TIRInterpreter(source).run(b,{},False)
        np.testing.assert_array_equal(b[0].values,[np.inf,-np.inf,8])
        from tir_mlir_checker.execution import Unsupported
        with self.assertRaises(Unsupported):
            TIRInterpreter(source.replace('T.infinity("float32")','T.infinity("int32")')).run(b,{},False)

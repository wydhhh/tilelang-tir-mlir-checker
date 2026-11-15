"""Generic operation tests, independent oracles and deliberately invalid variants."""
import unittest
import struct
import numpy as np
from tir_mlir_checker.execution import Buffer,Budget,Unsupported,SemanticFault,apply,cast,dtype,make_buffers
from tir_mlir_checker.tir_interpreter import TIRInterpreter
from tir_mlir_checker.mlir_interpreter import MLIRInterpreter


def tir(body):
    return '@I.ir_module\nclass Module:\n    @T.prim_func\n    def kernel(a: T.handle, out: T.handle):\n'+''.join('        '+line+'\n' for line in body.splitlines())

THREAD=tir('''a = T.match_buffer(a, (4,))
out = T.match_buffer(out, (4,))
tid = T.launch_thread("threadIdx.x", 4)
with T.block("root"):
    shared = T.alloc_buffer((4,), scope="shared")
    private = T.alloc_buffer((1,), scope="local")
    private[0] = a[tid]
    shared[tid] = private[0]
    T.tvm_storage_sync("shared")
    out[tid] = shared[(tid + 1) % 4]''')


class ExtensionTests(unittest.TestCase):
    def buffers(self,arrays,outputs):return make_buffers(arrays,outputs,False,Budget())
    def test_signed_floor_division(self):
        for a,b in [(-7,3),(7,-3),(-7,-3),(7,3)]:
            self.assertEqual(apply('floordiv',[np.int32(a),np.int32(b)],'i32'),a//b)
    def test_invalid_integer_division_is_not_a_pass(self):
        for a,b in [(1,0),(-2147483648,-1)]:
            with self.assertRaises(SemanticFault):apply('floordiv',[np.int32(a),np.int32(b)],'i32')
    def test_math_edges(self):
        self.assertEqual(apply('exp',[np.float32(0)],'f32'),1)
        self.assertEqual(apply('rsqrt',[np.float32(4)],'f32'),.5)
        self.assertTrue(np.isinf(apply('rsqrt',[np.float32(0)],'f32')))
        self.assertTrue(np.isnan(apply('rsqrt',[np.float32(-1)],'f32')))
        self.assertEqual(apply('sigmoid',[np.float32(0)],'f32'),.5)
    def test_all_cmpf_predicates_nan(self):
        for pred in ['oeq','one','olt','ole','ogt','oge','ueq','une','ult','ule','ugt','uge','ord','uno','true','false']:
            src='module {\n  func.func @k(%a: memref<1xf32>, %b: memref<1xi1>) {\n    %z = arith.constant 0 : index\n    %x = memref.load %a[%z] : memref<1xf32>\n    %r = arith.cmpf '+pred+', %x, %x : f32\n    memref.store %r, %b[%z] : memref<1xi1>\n    return\n  }\n}\n'
            b=self.buffers((np.array([np.nan],np.float32),np.empty(1,np.bool_)),[1]);MLIRInterpreter(src).run(b)
            self.assertEqual(bool(b[1].values[0]),pred.startswith('u') or pred=='true',pred)
    def test_f16_rounding_matches_python_struct(self):
        for x in [0.,-0.,1.,1.00048828125,1.00146484375,2**-24,2**-25,65504.]:
            actual=np.asarray(cast(np.float32(x),'f16')).tobytes()
            self.assertEqual(actual,struct.pack('<e',x))
    def test_bf16_all_finite_promotions_roundtrip_and_halfway(self):
        try:d=dtype('bf16')
        except Unsupported:self.skipTest('optional ml_dtypes missing')
        bits=np.arange(65536,dtype=np.uint16);a=bits.view(d)
        finite=np.isfinite(a)
        promoted=a.astype(np.float32).view(np.uint32)
        np.testing.assert_array_equal(promoted[finite],bits[finite].astype(np.uint32)<<16)
        np.testing.assert_array_equal(a.astype(np.float32).astype(d).view(np.uint16)[finite],bits[finite])
        for raw in [0x3f808000,0x3f818000,0x00008000,0x00018000,0x7f7fffff,0x80008000]:
            x=np.array(raw,np.uint32).view(np.float32)[()]
            expected=((raw+0x7fff+((raw>>16)&1))>>16)&0xffff
            self.assertEqual(np.asarray(cast(x,'bf16')).view(np.uint16),expected)
    def test_region_copy_rank_squeeze_and_padding(self):
        source=tir('''a = T.match_buffer(a, (1, 3))
out = T.match_buffer(out, (4,))
T.copy(T.region(a[0, 0], 1, 1, 4), T.region(out[0], 2, 4), disable_tma=T.bool(True))''')
        b=self.buffers((np.array([[1,2,3]],np.float32),np.empty(4,np.float32)),[1]);TIRInterpreter(source).run(b)
        np.testing.assert_array_equal(b[1].values,[1,2,3,0])
    def test_scalar_oob_is_not_padded(self):
        source=tir('a = T.match_buffer(a, (4,))\nout = T.match_buffer(out, (4,))\nout[0] = a[4]')
        with self.assertRaises(SemanticFault):TIRInterpreter(source).run(self.buffers((np.ones(4,np.float32),np.empty(4,np.float32)),[1]))
    def test_data_dependent_copy_is_numeric_only(self):
        source=tir('''a = T.match_buffer(a, (4,), "int32")
out = T.match_buffer(out, (1,), "int32")
T.copy(T.region(a[a[0]], 1, 1), T.region(out[0], 2, 1))''')
        arrays=(np.array([1,2,3,4],np.int32),np.empty(1,np.int32))
        numeric=make_buffers(arrays,(1,),False,Budget());TIRInterpreter(source).run(numeric)
        np.testing.assert_array_equal(numeric[1].values,[2])
        with self.assertRaisesRegex(Unsupported,'数据相关符号下标'):
            TIRInterpreter(source).run(make_buffers(arrays,(1,),True,Budget()),symbolic=True)
    def test_fill_subregion(self):
        source=tir('a = T.match_buffer(a, (4,))\nout = T.match_buffer(out, (4,))\nT.fill(out, 1)\nT.fill(T.region(out[1], 2, 2), 9)')
        b=self.buffers((np.ones(4,np.float32),np.empty(4,np.float32)),[1]);TIRInterpreter(source).run(b)
        np.testing.assert_array_equal(b[1].values,[1,9,9,1])
    def test_subview_shares_masks_writes_and_values(self):
        a=Buffer(np.empty((3,4),np.float32),initialized=False,symbolic=False)
        v=a.subview((1,0),(1,2),(1,2),(0,));v.write((1,),7,23)
        self.assertEqual(a.values[1,2],7);self.assertTrue(a.initialized[1,2]);self.assertEqual(a.last_lines[1,2],23)
        with self.assertRaises(SemanticFault):v.read((0,),24)
    def test_subview_alias_race_is_detected(self):
        a=Buffer(np.ones(4,np.float32),initialized=True,symbolic=False);v=a.subview((1,),(2,),(1,))
        with a.budget.iteration(1,0):a.write((1,),4,2)
        with a.budget.iteration(1,1):
            with self.assertRaises(SemanticFault):v.read((0,),3)
    def test_physical_private_shared_and_barrier(self):
        b=self.buffers((np.arange(4,dtype=np.float32),np.empty(4,np.float32)),[1]);TIRInterpreter(THREAD).run(b)
        np.testing.assert_array_equal(b[1].values,[1,2,3,0])
    def test_missing_barrier_rejected(self):
        with self.assertRaises(SemanticFault):TIRInterpreter(THREAD.replace('            T.tvm_storage_sync("shared")\n','')).run(self.buffers((np.arange(4,dtype=np.float32),np.empty(4,np.float32)),[1]))
    def test_divergent_barrier_rejected(self):
        changed=THREAD.replace('            T.tvm_storage_sync("shared")','            if tid > 0:\n                T.tvm_storage_sync("shared")')
        with self.assertRaises(SemanticFault):TIRInterpreter(changed).run(self.buffers((np.arange(4,dtype=np.float32),np.empty(4,np.float32)),[1]))
    def test_thread_write_race_rejected(self):
        with self.assertRaises(SemanticFault):TIRInterpreter(THREAD.replace('shared[tid] =','shared[0] =')).run(self.buffers((np.arange(4,dtype=np.float32),np.empty(4,np.float32)),[1]))
    def test_missing_fragment_metadata_is_explicit(self):
        changed=THREAD.replace('            out[tid] = shared[(tid + 1) % 4]','            for i in T.parallel(4, annotations={"parallel_loop_layout": metadata["tl.Fragment"][0]}):\n                out[i] = shared[i]')
        with self.assertRaisesRegex(Unsupported,'metadata'):TIRInterpreter(changed).run(self.buffers((np.arange(4,dtype=np.float32),np.empty(4,np.float32)),[1]))
    def test_unknown_pdl_sync_is_not_ignored(self):
        with self.assertRaisesRegex(Unsupported,'pdl_sync'):TIRInterpreter(THREAD.replace('T.tvm_storage_sync("shared")','T.pdl_sync()'))

class ProtocolAndDagTests(unittest.TestCase):
    def test_inout_preserves_initial_values_and_partial_updates(self):
        from tir_mlir_checker.semantics import coverage
        source=tir('a = T.match_buffer(a, (4,))\nout = T.match_buffer(out, (4,))\nout[0] = out[0] + a[0]')
        arrays=(np.ones(4,np.float32),np.full(4,3,np.float32))
        buffers=make_buffers(arrays,(1,),False,Budget(),inouts=(1,))
        TIRInterpreter(source).run(buffers)
        np.testing.assert_array_equal(buffers[1].values,[4,3,3,3])
        self.assertEqual(coverage(buffers[1])['status'],'match')
        with self.assertRaises(SemanticFault):TIRInterpreter(source).run(make_buffers(arrays,(1,),False,Budget()))
    def test_auxiliary_writable_workspace_is_explicit(self):
        a=np.zeros(1,np.float32)
        read_only=make_buffers((a,a.copy()),(1,),False,Budget())
        with self.assertRaises(SemanticFault):read_only[0].write((0,),1,4)
        writable=make_buffers((a,a.copy()),(1,),False,Budget(),inouts=(0,))
        writable[0].write((0,),1,4)
        self.assertEqual(writable[0].values[0],1)
    def test_dag_comparison_and_digest_do_not_expand_exponentially(self):
        from tir_mlir_checker.execution import Expr
        left=Expr('input',(0,(0,)),'f32');right=Expr('input',(0,(0,)),'f32')
        for _ in range(1500):
            left=Expr('add',(left,left),'f32');right=Expr('add',(right,right),'f32')
        self.assertEqual(left,right);self.assertEqual(left.fingerprint(),right.fingerprint())
        self.assertIn('truncated',str(left.data()))
        self.assertNotEqual(left,Expr('sub',right.args,'f32'))
    def test_bf16_random_rounding_independent_integer_oracle(self):
        try:d=dtype('bf16')
        except Unsupported:self.skipTest('optional ml_dtypes missing')
        raw=np.random.default_rng(41).integers(0,2**32,20000,dtype=np.uint32)
        raw=raw[(raw&0x7f800000)!=0x7f800000]
        rounded=((raw.astype(np.uint64)+0x7fff+((raw>>16)&1))>>16).astype(np.uint16)
        np.testing.assert_array_equal(raw.view(np.float32).astype(d).view(np.uint16),rounded)
    def test_allocation_budget(self):
        b=Budget(max_elements=1)
        for _ in range(8):Buffer(np.zeros(1,np.float32),initialized=True,symbolic=False,budget=b)
        with self.assertRaises(Unsupported):Buffer(np.zeros(1,np.float32),initialized=True,symbolic=False,budget=b)

class FragmentTests(unittest.TestCase):
    def test_complete_mapping_executes_once_per_logical_index(self):
        nodes=[{'type':'ffi.String','data':'tl.Fragment'},
               {'type':'ir.IntImm','data':{'value':2}}, {'type':'ffi.Array','data':[1,1]},
               {'type':'ffi.String','data':'_i'},{'type':'tir.Var','data':{'name':3}},
               {'type':'ffi.String','data':'_j'},{'type':'tir.Var','data':{'name':5}},
               {'type':'tir.Mul','data':{'a':4,'b':1}},{'type':'tir.Add','data':{'a':7,'b':6}},
               {'type':'ir.IntImm','data':{'value':1}},
               {'type':'tl.Fragment','data':{'input_size':2,'forward_thread':8,'forward_index':2,'replicate_size':9}},
               {'type':'ffi.Array','data':[10]},{'type':'ffi.Map','data':[0,11]}]
        metadata={'root_index':12,'nodes':nodes}
        changed=THREAD.replace('            out[tid] = shared[(tid + 1) % 4]',
            '            for i in T.parallel(2, annotations={"parallel_loop_layout": metadata["tl.Fragment"][0]}):\n                for j in T.parallel(2):\n                    out[i * 2 + j] = shared[(i * 2 + j + 1) % 4]')
        b=make_buffers((np.arange(4,dtype=np.float32),np.empty(4,np.float32)),(1,),False,Budget())
        TIRInterpreter(changed,metadata=metadata).run(b)
        np.testing.assert_array_equal(b[1].values,[1,2,3,0]);np.testing.assert_array_equal(b[1].writes,np.ones(4))
        nodes[9]['data']['value']=2
        with self.assertRaisesRegex(Unsupported,'replicate_size'):TIRInterpreter(changed,metadata=metadata).run(make_buffers((np.arange(4,dtype=np.float32),np.empty(4,np.float32)),(1,),False,Budget()))

class SerializationAndPolicyTests(unittest.TestCase):
    def test_float_predicate_expression_can_be_serialized(self):
        import json
        from tir_mlir_checker.execution import Expr,literal
        e=Expr('cmpf_olt',(literal(np.float32(0)),Expr('input',(0,(0,)),'f32')),'i1')
        json.dumps(e.data(),allow_nan=False)
        self.assertEqual(len(e.fingerprint()),64)
    def test_signed_remainder_truncates_toward_zero(self):
        for a,b,want in [(-7,3,-1),(7,-3,1),(-7,-3,-1),(7,3,1)]:
            self.assertEqual(apply('rem',[np.int32(a),np.int32(b)],'i32'),want)
    def test_per_output_comparison_contract(self):
        from tir_mlir_checker.cases import TestCase,Comparison,validate_case,compare
        arrays=tuple(np.zeros(1,np.float32) for _ in range(3))
        c=TestCase('dual',arrays,(1,2),(arrays[1],arrays[2]),output_comparisons={2:Comparison('tolerance',atol=.01)})
        validate_case(c)
        difference=np.array([.001],np.float32)
        self.assertFalse(compare(difference,arrays[1],c.comparison)['match'])
        self.assertTrue(compare(difference,arrays[2],c.output_comparisons[2])['match'])

class CommonScalarAndCopyTests(unittest.TestCase):
    def test_negf_preserves_zero_sign(self):
        src='module {\n  func.func @k(%a: memref<1xf32>, %b: memref<1xf32>) {\n    %i = arith.constant 0 : index\n    %x = memref.load %a[%i] : memref<1xf32>\n    %r = arith.negf %x : f32\n    memref.store %r, %b[%i] : memref<1xf32>\n    return\n  }\n}\n'
        for value in [0.,-0.,1.,float('inf')]:
            b=make_buffers((np.array([value],np.float32),np.empty(1,np.float32)),(1,),False,Budget());MLIRInterpreter(src).run(b)
            self.assertEqual(b[1].values.tobytes(),np.array([-value],np.float32).tobytes())
    def test_memref_copy_tracks_initialization(self):
        src='module {\n  func.func @k(%a: memref<2xf32>, %b: memref<2xf32>) {\n    memref.copy %a, %b : memref<2xf32> to memref<2xf32>\n    return\n  }\n}\n'
        b=make_buffers((np.array([1,2],np.float32),np.empty(2,np.float32)),(1,),False,Budget());MLIRInterpreter(src).run(b)
        np.testing.assert_array_equal(b[1].values,[1,2]);np.testing.assert_array_equal(b[1].writes,[1,1])
        b[0].initialized[1]=False
        with self.assertRaises(SemanticFault):MLIRInterpreter(src).run(b)
    def test_tir_min_max_finite_and_special_limits(self):
        src=tir('a = T.match_buffer(a, (4,))\nout = T.match_buffer(out, (4,))\nfor i in range(4):\n    out[i] = T.min(T.max(a[i], 0.0), 1.0)')
        b=make_buffers((np.array([-1,.3,2,0],np.float32),np.empty(4,np.float32)),(1,),False,Budget());TIRInterpreter(src).run(b)
        np.testing.assert_array_equal(b[1].values,np.array([0,.3,1,0],np.float32))
        with self.assertRaises(Unsupported):TIRInterpreter(src).run(make_buffers((np.array([np.nan,0,0,0],np.float32),np.empty(4,np.float32)),(1,),False,Budget()))

if __name__=='__main__':unittest.main()

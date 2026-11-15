"""Independent execution, non-Where fixture, and refreshed-hash mutations."""
import base64
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np=None

from tir_mlir_checker import check
from tir_mlir_checker.trace_io import STAGES

TOOL=Path(__file__).resolve().parents[1]
WHERE=TOOL/'tests/fixtures/traces/十算子-2026-11-11/Where'
CASE=TOOL/'tests/adapters/where_case.py'


def write_group(root,sources):
    root.mkdir()
    stages=[]
    for stage,source in zip(STAGES,sources):
        file=stage+('.tir' if stage in STAGES[:3] else '.mlir')
        raw=source.encode()
        (root/file).write_bytes(raw)
        stages.append({'id':stage,'file':file,'sha256':hashlib.sha256(raw).hexdigest(),
                       'bytes':len(raw),'lines':len(raw.splitlines())})
    (root/'manifest.json').write_text(json.dumps({'schema_version':1,'stages':stages}))
    return root


@unittest.skipIf(np is None,'execution extra requires NumPy')
class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='checker-execution-test-')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.sources=[(WHERE/(s+('.tir' if s in STAGES[:3] else '.mlir'))).read_text() for s in STAGES]
        self.short_case=self.root/'case.py'
        self.short_case.write_text(CASE.read_text().replace('def make_cases(seed):','def _all_cases(seed):')+
                                   '\ndef make_cases(seed):\n    yield from list(_all_cases(seed))[:2]\n')

    def execute_mutant(self,name,index,transform):
        sources=list(self.sources)
        changed=transform(sources[index])
        self.assertNotEqual(changed,sources[index],name)
        sources[index]=changed
        trace=write_group(self.root/name,sources)
        result=check(trace,dynamic=True,case=self.short_case,timeout=60)
        self.assertNotEqual(result['tool_status'],'input_error',result['diagnostics'])
        return result

    def test_where_full_five_stage_nine_case_bitwise_comparison(self):
        result=check(WHERE,dynamic=True,case=CASE,timeout=60,out_dir=self.root/'report')
        self.assertEqual(result['tool_status'],'ok',result['dynamic'])
        self.assertEqual(result['semantic_conclusion'],'bounded_checks_passed')
        self.assertEqual(result['dynamic']['case_count'],9)
        self.assertEqual(len(result['dynamic']['tests']),45)
        self.assertTrue(all(t['bitwise_equal_elements']==2048 for t in result['dynamic']['tests']))
        expressions=[s for s in result['dynamic']['symbolic'] if s.get('kind')=='expression']
        self.assertEqual(len(expressions),4)
        self.assertTrue(all(s['status']=='match' and s['equal_expressions']==2048 for s in expressions))
        with zipfile.ZipFile(result['artifacts']['evidence']) as z:
            self.assertIsNone(z.testzip())
            arrays=np.load(io.BytesIO(z.read('arrays.npz')),allow_pickle=False)
            self.assertEqual(arrays['case0_arg0'].dtype,np.uint8)
            self.assertTrue(np.array_equal(arrays['case0_03_mlir_linalg_out3'],arrays['case0_expected3']))
            saved=json.loads(z.read('result.json'))
            self.assertNotIn('_evidence',saved)

    def test_where_condition_reversal(self):
        result=self.execute_mutant('condition',3,lambda s:s.replace('arith.cmpi ne, %23','arith.cmpi eq, %23'))
        self.assertEqual(result['tool_status'],'mismatch',result['dynamic'])
        self.assertEqual(result['semantic_conclusion'],'bounded_counterexample')
        failure=next(t for t in result['dynamic']['tests'] if t['status']=='mismatch')
        self.assertEqual(failure['first_difference']['index'],[0])
        expr=next(s for s in result['dynamic']['symbolic'] if s.get('from')==STAGES[2])
        self.assertEqual(expr['status'],'uncertain')
        self.assertNotEqual(expr['first_difference']['left'],expr['first_difference']['right'])

    def test_where_branch_swap(self):
        result=self.execute_mutant('swap',3,lambda s:s.replace('memref.load %alloca_0[%30]','memref.load %TEMP[%30]').replace('memref.load %alloca[%30]','memref.load %alloca_0[%30]').replace('%TEMP','%alloca'))
        self.assertEqual(result['tool_status'],'mismatch',result['dynamic'])

    def test_global_source_substitution(self):
        result=self.execute_mutant('source',3,lambda s:s.replace('memref.load %arg1[%13]','memref.load %arg1[%c0]'))
        self.assertEqual(result['tool_status'],'mismatch',result['dynamic'])
        expr=next(s for s in result['dynamic']['symbolic'] if s.get('from')==STAGES[2])
        self.assertEqual(expr['first_difference']['index'],[1])
        self.assertIn('input',str(expr['first_difference']['right']))

    def test_wrong_block_offset(self):
        result=self.execute_mutant('offset',3,lambda s:s.replace('arith.constant 512 : i32','arith.constant 256 : i32'))
        self.assertEqual(result['tool_status'],'mismatch',result['dynamic'])

    def test_short_output_loop(self):
        result=self.execute_mutant('short',3,lambda s:s.replace('scf.for %arg7 = %c0_18 to %9 step %c1_20','scf.for %arg7 = %c0_18 to %c1 step %c1_20'))
        self.assertEqual(result['tool_status'],'mismatch',result['dynamic'])
        cov=next(s for s in result['dynamic']['symbolic'] if s.get('stage')==STAGES[3] and s.get('kind')=='coverage')
        self.assertEqual(cov['unwritten'],2044)

    def test_missing_store_and_reproducible_arrays(self):
        result=self.execute_mutant('store',3,lambda s:s.replace('memref.store %24, %arg3[%13] : memref<2048xf32>',''))
        self.assertEqual(result['tool_status'],'mismatch',result['dynamic'])
        stream=base64.b64decode(result['_evidence']['arrays.npz'])
        arrays=np.load(io.BytesIO(stream),allow_pickle=False)
        self.assertTrue(np.all(arrays['case1_arg0']==1))
        self.assertEqual(result['dynamic']['seed'],0)
        self.assertEqual(result['dynamic']['stage_sha256'][STAGES[3]],hashlib.sha256((self.root/'store/03_mlir_linalg.mlir').read_bytes()).hexdigest())

    def test_out_of_bounds_reports_memory_fault(self):
        result=self.execute_mutant('oob',3,lambda s:s.replace('memref.load %arg1[%13]','memref.load %arg1[%c2048_i32]'))
        self.assertEqual(result['tool_status'],'mismatch',result['dynamic'])
        self.assertTrue(any('越界' in t.get('reason','') for t in result['dynamic']['tests']))

    def test_unknown_operation_in_unreached_branch_is_unsupported(self):
        result=self.execute_mutant('unknown',2,lambda s:s+'\n        if False:\n            T.unknown_call()\n')
        self.assertEqual(result['tool_status'],'unsupported',result['dynamic'])
        self.assertEqual(result['dynamic']['stage_support'][STAGES[2]]['status'],'unsupported')
        self.assertFalse(result['dynamic']['has_mismatch'])

    def test_physical_thread_dependent_code_is_unsupported(self):
        result=self.execute_mutant('threads',2,lambda s:s.replace('for i in T.parallel(128):','for i in T.parallel(tx + 1):'))
        self.assertEqual(result['tool_status'],'unsupported',result['dynamic'])
        self.assertTrue(any('物理线程轴' in s.get('reason','') for s in result['dynamic']['symbolic']))

    def test_logical_parallel_write_conflict(self):
        result=self.execute_mutant('race',3,lambda s:s.replace('memref.store %25, %alloca_0[%17]','memref.store %25, %alloca_0[%c0]'))
        self.assertEqual(result['tool_status'],'mismatch',result['dynamic'])
        self.assertTrue(any('parallel' in t.get('reason','') for t in result['dynamic']['tests']))

    def test_semantic_only_does_not_claim_numeric_execution(self):
        result=check(WHERE,semantic=True,case=self.short_case,timeout=60)
        self.assertEqual(result['tool_status'],'ok',result['dynamic'])
        self.assertFalse(result['dynamic']['executed'])
        self.assertEqual(result['dynamic']['tests'],[])
        self.assertEqual(result['semantic_conclusion'],'bounded_expressions_match')

    def test_symbolic_potential_fault_is_not_a_concrete_counterexample(self):
        sources=list(self.sources)
        before='T.if_then_else(c_reg[i * 4 + j] != T.uint8(0), x_reg[i * 4 + j], y_reg[i * 4 + j])'
        after='T.if_then_else(x_reg[0] != x_reg[0], x_reg[512], x_reg[i * 4 + j])'
        self.assertIn(before,sources[2])
        sources[2]=sources[2].replace(before,after)
        trace=write_group(self.root/'potential_fault',sources)
        result=check(trace,semantic=True,case=self.short_case,timeout=60)
        self.assertEqual(result['summary']['mismatch'],0)
        self.assertFalse(result['dynamic']['has_mismatch'])
        self.assertEqual(result['semantic_conclusion'],'bounded_expressions_unresolved')

    def test_core_rejects_removed_native_api(self):
        with self.assertRaises(TypeError):
            check(WHERE,dynamic=True,case=self.short_case,backend='native')

    def test_non_where_two_dimensional_multi_output_adapter(self):
        tir='''@I.ir_module
class Module:
    @T.prim_func
    def main(ha: T.handle, hb: T.handle, hs: T.handle, ht: T.handle):
        a = T.match_buffer(ha, (3, 5))
        b = T.match_buffer(hb, (3, 5))
        s = T.match_buffer(hs, (3, 5))
        t = T.match_buffer(ht, (5, 3))
        for i in T.serial(3):
            for j in T.parallel(5):
                s[i, j] = a[i, j] + b[i, j]
                t[j, i] = a[i, j]
'''
        mlir='''module {
  func.func @main(%a: memref<3x5xf32>, %b: memref<3x5xf32>, %s: memref<3x5xf32>, %t: memref<5x3xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c5 = arith.constant 5 : index
    scf.for %i = %c0 to %c3 step %c1 {
      scf.for %j = %c0 to %c5 step %c1 {
        %x = memref.load %a[%i, %j] : memref<3x5xf32>
        %y = memref.load %b[%i, %j] : memref<3x5xf32>
        %z = arith.addf %x, %y : f32
        memref.store %z, %s[%i, %j] : memref<3x5xf32>
        memref.store %x, %t[%j, %i] : memref<5x3xf32>
      }
    }
    return
  }
}
'''
        trace=write_group(self.root/'sum_transpose',[tir]*3+[mlir]*2)
        adapter=self.root/'other_case.py'
        adapter.write_text('''import numpy as np
from tir_mlir_checker.cases import TestCase
def make_cases(seed):
    rng=np.random.default_rng(seed)
    a=rng.normal(size=(3,5)).astype(np.float32)
    b=rng.normal(size=(3,5)).astype(np.float32)
    yield TestCase('sum_and_transpose',(a,b,np.empty_like(a),np.empty((5,3),np.float32)),(2,3),(a+b,a.T.copy()))
''')
        result=check(trace,dynamic=True,case=adapter,timeout=60)
        self.assertEqual(result['tool_status'],'ok',result['dynamic'])
        self.assertEqual(len(result['dynamic']['tests']),10)
        self.assertEqual(result['semantic_conclusion'],'bounded_checks_passed')

    def test_comparison_policies_nan_infinity_signed_zero(self):
        from tir_mlir_checker.cases import Comparison,compare
        a=np.array([0.,np.inf,np.nan],np.float32)
        b=np.array([-0.,np.inf,np.nan],np.float32)
        self.assertFalse(compare(a,b,Comparison())['match'])
        self.assertFalse(compare(a,b,Comparison(mode='tolerance',equal_nan=True))['match'])
        self.assertTrue(compare(a,b,Comparison(mode='tolerance',equal_nan=True,signed_zero_equal=True))['match'])
        self.assertFalse(compare(np.array([np.inf]),np.array([-np.inf]),Comparison(mode='tolerance'))['match'])


if __name__=='__main__':
    unittest.main()

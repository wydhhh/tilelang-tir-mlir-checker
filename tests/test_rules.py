"""Automatic diagnostics: correct, faulty, legal rewrite and unsupported cases."""
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

from tir_mlir_checker import check
from tir_mlir_checker.rules import tir_inventory
from tir_mlir_checker.trace_io import STAGES
from test_execution import write_group, WHERE, CASE, TOOL


COPY_TIR = '''@I.ir_module
class Module:
    @T.prim_func
    def main(ha: T.handle, hb: T.handle):
        a = T.match_buffer(ha, (2, 3))
        b = T.match_buffer(hb, (2, 3))
        for i in T.serial(2):
            for j in T.serial(3):
                b[i, j] = a[i, j]
'''
COPY_MLIR = '''module {
  func.func @main(%a: memref<2x3xf32>, %b: memref<2x3xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    scf.for %i = %c0 to %c2 step %c1 {
      scf.for %j = %c0 to %c3 step %c1 {
        %x = memref.load %a[%i, %j] : memref<2x3xf32>
        memref.store %x, %b[%i, %j] : memref<2x3xf32>
      }
    }
    return
  }
}
'''


class StaticRuleTests(unittest.TestCase):
    def test_collects_multiple_unknowns_in_unreached_code(self):
        source = COPY_TIR + '        if False:\n            T.foo()\n            T.bar()\n'
        _, findings = tir_inventory(source, STAGES[2])
        self.assertEqual([f['details']['operation'] for f in findings], ['T.foo','T.bar'])
        self.assertTrue(all(f['status']=='unsupported' and f['locations'][0]['line'] for f in findings))

    def test_reduction_and_conversion_facts_keep_original_context(self):
        source = COPY_TIR + '        T.reduce(a[0:2, 0:3], b[0:2, 0], "sum", 1, T.bool(False))\n'
        facts, gaps = tir_inventory(source, STAGES[2])
        self.assertFalse(gaps)
        self.assertEqual(facts['reductions'][0]['arguments'][3:], ['1', 'T.bool(False)'])
        self.assertEqual(facts['loops'][1]['enclosing_loops'], [7])

    def test_metadata_is_not_executed_or_treated_as_calls(self):
        source = COPY_TIR.replace('        for i', '        T.func_attr({"target": T.target({"x": "bf16"})})\n        for i', 1)
        _, gaps = tir_inventory(source, STAGES[2])
        self.assertFalse(gaps)

    def test_low_precision_is_recorded_as_optional_dependency(self):
        source = COPY_TIR.replace('(2, 3))', '(2, 3), "bfloat16")')
        facts, gaps = tir_inventory(source, STAGES[2])
        self.assertFalse(gaps)
        self.assertEqual(len(facts['low_precision']),2)

    def test_mlir_collects_all_named_gaps_and_explicit_layout(self):
        from tir_mlir_checker.analysis import mlir_facts
        from tir_mlir_checker.rules import mlir_inventory
        source=COPY_MLIR.replace('memref<2x3xf32>', 'memref<2x3xbf16, strided<[3, 1]>>').replace(
            '    return', '    %p = math.exp %x : bf16\n    %q = math.rsqrt %p : bf16\n    return')
        _, gaps=mlir_inventory(source,STAGES[3],mlir_facts(source))
        self.assertEqual({g['details'].get('operation') for g in gaps if g['rule_id']=='CAP-MLIR-OP'}, set())
        self.assertFalse(gaps)
        affine=source.replace('strided<[3, 1]>', 'affine_map<(d0,d1)->(d1,d0)>')
        _,unknown=mlir_inventory(affine,STAGES[3],mlir_facts(affine))
        self.assertTrue(any(g['rule_id']=='CAP-LAYOUT' for g in unknown))


@unittest.skipIf(np is None, 'execution extra requires NumPy')
class AutomaticDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='checker-auto-rule-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.case = self.root/'case.py'
        self.case.write_text('''import numpy as np
from tir_mlir_checker.cases import TestCase
def make_cases(seed):
    a=np.arange(6,dtype=np.float32).reshape(2,3)+1
    yield TestCase('distinct',(a,np.empty_like(a)),(1,),(a.copy(),))
''')

    def run_group(self, tir=COPY_TIR, mlir=COPY_MLIR, **kwargs):
        trace=write_group(self.root/'trace',[COPY_TIR,COPY_TIR,tir,mlir,COPY_MLIR])
        return check(trace, case=self.case, dynamic=True, timeout=60, **kwargs)

    def rule(self, result, code, status='mismatch'):
        return next(f for f in result['findings'] if f['rule_id']==code and f['status']==status)

    def test_correct_copy_and_full_report_contract(self):
        result=self.run_group(out_dir=self.root/'report')
        self.assertEqual(result['tool_status'],'ok')
        self.assertFalse(result['findings'])
        with zipfile.ZipFile(result['artifacts']['evidence']) as z:
            saved=json.loads(z.read('result.json'))
        self.assertEqual(saved['rule_facts'],result['rule_facts'])
        self.assertEqual(saved['schema_version'],2)
        self.assertIn('问题定位与 Agent 复核',Path(result['artifacts']['report']).read_text())

    def test_oob_has_buffer_loop_value_line_and_original_excerpt(self):
        result=self.run_group(mlir=COPY_MLIR.replace('memref.load %a[%i, %j]','memref.load %a[%i, %c3]'))
        f=self.rule(result,'MEM-OUT-OF-BOUNDS')
        self.assertEqual(f['details']['index'],[0,3])
        self.assertEqual(f['details']['shape'],[2,3])
        self.assertEqual(f['details']['buffer'],'%a')
        self.assertTrue(any(c.get('variable')=='%j' and c.get('value')==0 for c in f['details']['context']))
        self.assertIn('memref.load',f['locations'][0]['excerpt'])
        self.assertEqual(f['method'],'bounded_execution')

    def test_uninitialized_read_reports_allocation_and_read_sites(self):
        mlir=COPY_MLIR.replace('    %c0 =', '    %tmp = memref.alloca() : memref<2x3xf32>\n    %c0 =',1)
        mlir=mlir.replace('memref.load %a[%i, %j]','memref.load %tmp[%i, %j]')
        f=self.rule(self.run_group(mlir=mlir),'MEM-UNINITIALIZED')
        self.assertEqual(f['details']['allocation_line'],3)
        self.assertEqual(len(f['locations']),2)

    def test_readonly_write_reports_argument(self):
        result=self.run_group(tir=COPY_TIR.replace('b[i, j] = a[i, j]','a[i, j] = T.float32(0)\n                b[i, j] = a[i, j]'))
        f=self.rule(result,'MEM-READONLY-WRITE')
        self.assertEqual(f['details']['parameter'],0)

    def test_missing_output_and_short_loop_report_first_hole(self):
        result=self.run_group(mlir=COPY_MLIR.replace('scf.for %j = %c0 to %c3','scf.for %j = %c0 to %c2'))
        f=self.rule(result,'MEM-OUTPUT-UNWRITTEN')
        self.assertEqual(f['details']['unwritten'],2)
        self.assertEqual(f['details']['first_unwritten_index'],[0,2])

    def test_wrong_source_retains_full_copy_chain_and_numeric_witness(self):
        result=self.run_group(mlir=COPY_MLIR.replace('memref.load %a[%i, %j]','memref.load %a[%i, %c0]'))
        self.rule(result,'FLOW-SOURCE-DIFFERENCE','uncertain')
        f=self.rule(result,'NUM-OUTPUT-DIFFERENCE')
        self.assertEqual(f['details']['first_difference']['index'],[0,1])
        self.assertTrue(any(c['buffer']=='%a' and c['action']=='read' for c in f['details']['source_chain']))

    def test_repeated_sequential_writes_are_not_automatically_errors(self):
        mlir=COPY_MLIR.replace('        memref.store %x, %b[%i, %j] : memref<2x3xf32>',
                              '        memref.store %x, %b[%i, %j] : memref<2x3xf32>\n        memref.store %x, %b[%i, %j] : memref<2x3xf32>')
        self.assertEqual(self.run_group(mlir=mlir)['tool_status'],'ok')

    def test_guarded_tail_does_not_report_unreachable_oob(self):
        source=COPY_TIR.replace('T.serial(3)', 'T.serial(4)').replace('                b[i, j]', '                if j < 3:\n                    b[i, j]')
        result=self.run_group(tir=source)
        self.assertEqual(result['tool_status'],'ok',result['findings'])

    def test_unsupported_stage_does_not_suppress_independent_stage_diagnostics(self):
        tir=COPY_TIR+'        if False:\n            T.foo()\n            T.bar()\n'
        result=self.run_group(tir=tir,mlir=COPY_MLIR.replace('memref.load %a[%i, %j]','memref.load %a[%c3, %j]'))
        self.rule(result,'MEM-OUT-OF-BOUNDS')
        names={f['details'].get('operation') for f in result['findings'] if f['rule_id']=='CAP-TIR-CALL'}
        self.assertEqual(names,{'T.foo','T.bar'})

    def test_parameter_contract_difference_stays_uncertain(self):
        result=self.run_group(mlir=COPY_MLIR.replace('memref<2x3xf32>','memref<3x2xf32>'))
        self.rule(result,'CONTRACT-SIGNATURE','uncertain')
        self.assertFalse(any(f['rule_id']=='CONTRACT-SIGNATURE' and f['status']=='mismatch' for f in result['findings']))

    def test_precision_change_has_conversion_sites_and_numeric_counterexample(self):
        tir=COPY_TIR.replace('(2, 3))','(2, 3), "float64")')
        mlir=COPY_MLIR.replace('f32','f64').replace('        memref.store %x,',
            '        %narrow = arith.truncf %x : f64 to f32\n        %wide = arith.extf %narrow : f32 to f64\n        memref.store %wide,')
        trace=write_group(self.root/'precision',[tir]*3+[mlir,COPY_MLIR.replace('f32','f64')])
        self.case.write_text(self.case.read_text().replace('np.float32','np.float64').replace('+1','+16777217'))
        result=check(trace,case=self.case,dynamic=True)
        self.rule(result,'PRECISION-CONVERSION-DIFFERENCE','uncertain')
        self.rule(result,'NUM-OUTPUT-DIFFERENCE')
        self.assertEqual(len(result['rule_facts'][STAGES[3]]['conversions']),2)
        f=self.rule(result,'PRECISION-CONVERSION-DIFFERENCE','uncertain')
        self.assertTrue(any('arith.truncf' in p.get('excerpt','') for p in f['locations']))

    def test_parallel_conflict_has_both_access_locations(self):
        tir=COPY_TIR.replace('T.serial(3)','T.parallel(3)').replace('b[i, j] =','b[i, 0] =')
        f=self.rule(self.run_group(tir=tir),'PARALLEL-CONFLICT')
        self.assertEqual(f['details']['previous_iteration'],0)
        self.assertEqual(f['details']['current_iteration'],1)
        self.assertIsInstance(f['details']['conflicting_line'],int)


@unittest.skipIf(np is None, 'execution extra requires NumPy')
class ReductionRuleTests(unittest.TestCase):
    def setUp(self):
        from tir_mlir_checker.execution import Budget, make_buffers
        from tir_mlir_checker.tir_interpreter import TIRInterpreter
        self.make_buffers,self.Budget,self.Engine=make_buffers,Budget,TIRInterpreter

    def execute(self, x, *, kind='sum', axis=1, clear=True, seed=0, initialized=True, symbolic=False):
        shape=x.shape
        outshape=shape[:axis]+shape[axis+1:]
        y=np.empty(outshape,dtype=x.dtype)
        prefix='        T.fill(out, T.float32('+str(seed)+'))\n' if initialized else ''
        source='''@I.ir_module
class Module:
    @T.prim_func
    def main(hx: T.handle, ho: T.handle):
        x = T.match_buffer(hx, SHAPE)
        out = T.match_buffer(ho, OUTSHAPE)
PREFIX        T.reduce(x, out, KIND, AXIS, T.bool(CLEAR))
'''.replace('OUTSHAPE',repr(outshape)).replace('SHAPE',repr(shape)).replace('PREFIX',prefix).replace('KIND',repr(kind)).replace('AXIS',str(axis)).replace('CLEAR',repr(clear))
        buffers=self.make_buffers((x,y),(1,),symbolic,self.Budget())
        info=self.Engine(source).run(buffers,symbolic=symbolic)
        return buffers[1].values,info

    def test_sum_axes_clear_and_preserve_nonzero_initial(self):
        x=np.arange(24,dtype=np.float32).reshape(2,3,4)
        for axis in range(3):
            with self.subTest(axis=axis):
                actual,info=self.execute(x,axis=axis,clear=False,seed=13)
                np.testing.assert_array_equal(actual,x.sum(axis=axis)+13)
                self.assertFalse(info['reductions'][0]['clear'])
                cleared,_=self.execute(x,axis=axis,clear=True,seed=13)
                np.testing.assert_array_equal(cleared,x.sum(axis=axis))

    def test_max_min_use_correct_identity_and_axis(self):
        x=-np.arange(1,25,dtype=np.float32).reshape(2,3,4)
        for kind in ['max','min']:
            actual,_=self.execute(x,kind=kind,axis=1,seed=999)
            np.testing.assert_array_equal(actual,getattr(x,kind)(axis=1))

    def test_clear_false_without_initialization_is_memory_error(self):
        from tir_mlir_checker.execution import SemanticFault
        with self.assertRaises(SemanticFault) as cm:
            self.execute(np.ones((2,3),np.float32),clear=False,initialized=False)
        self.assertEqual(cm.exception.rule_id,'MEM-UNINITIALIZED')

    def test_max_nan_is_unsupported_not_silent_pass(self):
        from tir_mlir_checker.execution import Unsupported
        with self.assertRaises(Unsupported):
            self.execute(np.array([[1,np.nan]],np.float32),kind='max')

    def test_max_symbolic_does_not_assume_special_values_are_absent(self):
        from tir_mlir_checker.execution import Unsupported
        with self.assertRaises(Unsupported):
            self.execute(np.ones((2,3),np.float32),kind='max',symbolic=True)

    def test_real_sum_five_stage_and_wrong_initialization_axis(self):
        sources=[(TOOL/'tests/fixtures/traces/十算子-2026-11-11/Sum'/(s+('.tir' if s in STAGES[:3] else '.mlir'))).read_text() for s in STAGES]
        with tempfile.TemporaryDirectory(prefix='checker-sum-rules-') as d:
            root=Path(d)
            original=check(TOOL/'tests/fixtures/traces/十算子-2026-11-11/Sum',case=TOOL/'tests/adapters/sum_case.py',dynamic=True)
            self.assertEqual(original['tool_status'],'ok',original['findings'])
            self.assertEqual(len(original['dynamic']['tests']),25)
            self.assertTrue(all(t['status']=='match' for t in original['dynamic']['tests']))
            # Preserve the legal expression difference: x0+0 vs 0+x0.
            self.assertEqual(original['semantic_conclusion'],'samples_passed_symbolic_unresolved')
            changed=list(sources)
            changed[2]=changed[2].replace('T.reduce(x_f32', 'T.fill(acc, T.float32(17))\n            T.reduce(x_f32').replace('T.bool(True))','T.bool(False))')
            trace=write_group(root/'wrong_initial',changed)
            result=check(trace,case=TOOL/'tests/adapters/sum_case.py',dynamic=True)
            self.assertTrue(any(f['rule_id']=='NUM-OUTPUT-DIFFERENCE' and f['status']=='mismatch' for f in result['findings']))
            self.assertTrue(any(f['rule_id']=='REDUCE-CONTRACT-DIFFERENCE' for f in result['findings']))

    def test_wrong_axis_in_square_input_has_numeric_counterexample(self):
        tir='''@I.ir_module
class Module:
    @T.prim_func
    def main(ha: T.handle, hb: T.handle):
        a = T.match_buffer(ha, (2, 2))
        b = T.match_buffer(hb, (2,))
        T.reduce(a, b, "sum", 1, T.bool(True))
'''
        mlir='''module {
  func.func @main(%a: memref<2x2xf32>, %b: memref<2xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %i = %c0 to %c2 step %c1 {
      %x = memref.load %a[%i, %c0] : memref<2x2xf32>
      %y = memref.load %a[%i, %c1] : memref<2x2xf32>
      %z = arith.addf %x, %y : f32
      memref.store %z, %b[%i] : memref<2xf32>
    }
    return
  }
}
'''
        with tempfile.TemporaryDirectory(prefix='checker-axis-') as d:
            root=Path(d)
            trace=write_group(root/'trace',[tir,tir,tir.replace('"sum", 1','"sum", 0'),mlir,mlir])
            case=root/'case.py'
            case.write_text('''import numpy as np
from tir_mlir_checker.cases import TestCase
def make_cases(seed):
    x=np.array([[1,3],[10,50]],np.float32)
    yield TestCase('square_axis',(x,np.empty(2,np.float32)),(1,),(x.sum(axis=1),))
''')
            result=check(trace,case=case,dynamic=True)
            self.assertEqual(result['tool_status'],'mismatch')
            f=next(f for f in result['findings'] if f['rule_id']=='NUM-OUTPUT-DIFFERENCE')
            self.assertEqual(f['stage'],STAGES[2])
            self.assertEqual(f['details']['first_difference']['index'],[0])


if __name__=='__main__':
    unittest.main()

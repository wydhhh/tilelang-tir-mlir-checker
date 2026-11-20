"""Real official MLIR tests run in the configured compiler environment."""
import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tests.support.timeouts import regression_timeout
from tir_mlir_checker import check
from tir_mlir_checker.trace_io import STAGES

ROOT=Path(__file__).resolve().parents[1]
FIXTURES=ROOT/'tests/fixtures/official_tir'
HAS_MLIR=importlib.util.find_spec('mlir') is not None
HAS_TVM=importlib.util.find_spec('tilelang') is not None

COPY='''module {
  func.func @main(%x: memref<4xf32>, %y: memref<4xf32>) {
    %zero = arith.constant 0 : index
    %four = arith.constant 4 : index
    %one = arith.constant 1 : index
    %c = arith.constant 1.0 : f32
    %tmp = memref.alloca() : memref<4xf32>
    linalg.fill ins(%c : f32) outs(%tmp : memref<4xf32>)
    scf.parallel (%i) = (%zero) to (%four) step (%one) {
      %a = memref.load %x[%i] : memref<4xf32>
      %b = memref.load %tmp[%i] : memref<4xf32>
      %v = arith.addf %a, %b : f32
      memref.store %v, %tmp[%i] : memref<4xf32>
      scf.reduce
    }
    memref.copy %tmp, %y : memref<4xf32> to memref<4xf32>
    return
  }
}'''

SUM='''module {
  func.func @sum(%x: memref<4x4xf32>, %y: memref<4xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %zero = arith.constant 0.0 : f32
    %acc = memref.alloca() : memref<4xf32>
    linalg.fill ins(%zero : f32) outs(%acc : memref<4xf32>)
    scf.for %i = %c0 to %c4 step %c1 {
      %base = memref.load %acc[%i] : memref<4xf32>
      %sum = scf.for %j = %c0 to %c4 step %c1 iter_args(%total = %base) -> (f32) {
        %v = memref.load %x[%i, %j] : memref<4x4xf32>
        %next = arith.addf %total, %v : f32
        scf.yield %next : f32
      }
      memref.store %sum, %y[%i] : memref<4xf32>
    }
    return
  }
}'''


class MLIRInputTests(unittest.TestCase):
    def test_configuration_errors_are_not_semantic_errors(self):
        for kwargs in ({'mlir_backend':'guess'},{'ir_python':'python'},
                       {'mlir_backend':'official','tir_backend':'tvm','ir_python':'a','tir_python':'b'}):
            self.assertEqual(check(FIXTURES/'where',**kwargs)['tool_status'],'input_error')
        result=check(FIXTURES/'where',mlir_backend='official',ir_python='/nonexistent/checker-python')
        self.assertEqual(result['tool_status'],'unsupported')
        self.assertEqual(result['semantic_conclusion'],'not_established')

    @unittest.skipIf(HAS_MLIR,'requires the lightweight host without MLIR')
    def test_missing_mlir_is_not_text_fallback(self):
        result=check(FIXTURES/'where',mlir_backend='official')
        self.assertEqual(result['tool_status'],'unsupported')
        self.assertEqual(len(result['mlir_python_verification']['runs']),2)
        self.assertTrue(all(r['failure_kind']=='dependency_unavailable' for r in result['mlir_python_verification']['runs']))


@unittest.skipUnless(HAS_MLIR,'requires configured official MLIR Python bindings')
class OfficialMLIRTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def group(self,name,source,second=None):
        trace=self.root/name;shutil.copytree(FIXTURES/'where',trace)
        manifest=json.loads((trace/'manifest.json').read_text())
        for record,text in zip(manifest['stages'][3:],(source,source if second is None else second)):
            raw=text.encode();(trace/record['file']).write_bytes(raw)
            record.update(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),lines=len(raw.splitlines()))
        (trace/'manifest.json').write_text(json.dumps(manifest))
        return trace

    def case(self,name,shape=(4,),reference='x+np.float32(1)'):
        file=self.root/(name+'.py')
        file.write_text('''import numpy as np
from tir_mlir_checker.cases import TestCase,Comparison
def make_cases(seed):
    x=np.arange('''+str(__import__('math').prod(shape))+''',dtype="float32").reshape('''+repr(shape)+''')+np.float32(.12345)
    expected='''+reference+'''
    yield TestCase("distinct",(x,np.empty_like(expected)),(1,),(expected,),comparison=Comparison("bitwise"))
''')
        return file

    def execute(self,trace,case,**kwargs):
        return check(trace,mlir_backend='official',dynamic=True,case=case,stages=STAGES[3:],timeout=regression_timeout(120),**kwargs)

    def test_official_static_check_is_separate_from_execution(self):
        result=check(FIXTURES/'where',mlir_backend='official',out_dir=self.root/'report')
        self.assertEqual(result['tool_status'],'ok',result)
        self.assertEqual(result['mlir_python_verification']['status'],'passed')
        self.assertEqual(result['dynamic']['status'],'not_requested')
        self.assertEqual(result['semantic_conclusion'],'not_established')
        self.assertTrue(all(result['facts'][s]['official_object_verified'] for s in STAGES[3:]))

    def test_copy_fill_parallel_and_generic_assembly(self):
        case=self.case('copy')
        result=self.execute(self.group('normal',COPY),case)
        self.assertEqual(result['tool_status'],'ok',result['dynamic'])
        self.assertEqual(len(result['dynamic']['tests']),2)
        import mlir.ir as ir
        with ir.Context() as ctx:
            ctx.load_all_available_dialects()
            generic=ir.Module.parse(COPY).operation.get_asm(print_generic_op_form=True)
        result=self.execute(self.group('generic','# comment\n'.replace('#','//')+generic),case)
        self.assertEqual(result['tool_status'],'ok',result['dynamic'])
        self.assertTrue(all(t['backend']=='mlir_operation_objects' for t in result['dynamic']['tests']))

    def test_five_deliberate_errors_produce_counterexamples(self):
        reference='np.add.reduce(x,axis=1,dtype=np.float32)'
        # Independent sequential reference, preserving the intended f32 order.
        case=self.case('sum',(4,4),reference)
        text=case.read_text().replace('expected='+reference,
            'expected=np.zeros(4,dtype="float32")\n    for j in range(4): expected=np.add(expected,x[:,j],dtype=np.float32)')
        case.write_text(text)
        valid=self.execute(self.group('valid',SUM),case)
        self.assertEqual(valid['tool_status'],'ok',valid['dynamic'])
        changes={
            'index':SUM.replace('%x[%i, %j]','%x[%i, %c0]'),
            'missing_write':SUM.replace('memref.store %sum, %y[%i] : memref<4xf32>',''),
            'initial_value':SUM.replace('arith.constant 0.0 : f32','arith.constant 1.0 : f32'),
            'early_precision':SUM.replace('scf.yield %next : f32',
                '%rough = arith.truncf %next : f32 to bf16\n        %up = arith.extf %rough : bf16 to f32\n        scf.yield %up : f32'),
            'uninitialized':SUM.replace('linalg.fill ins(%zero : f32) outs(%acc : memref<4xf32>)',''),
        }
        for name,source in changes.items():
            with self.subTest(name=name):
                result=self.execute(self.group(name,source),case)
                self.assertEqual(result['mlir_python_verification']['status'],'passed',result)
                self.assertEqual(result['tool_status'],'mismatch',result['dynamic'])
                self.assertEqual(result['semantic_conclusion'],'bounded_counterexample')
                self.assertTrue(any(f['rule_id'] in {'NUM-OUTPUT-DIFFERENCE','MEM-OUTPUT-UNWRITTEN','MEM-UNINITIALIZED'} for f in result['findings']))

    def test_out_of_bounds_and_explicit_locations(self):
        case=self.case('copy')
        bad=COPY.replace('%x[%i]','%x[%four]')
        result=self.execute(self.group('bounds',bad),case)
        self.assertEqual(result['tool_status'],'mismatch')
        fault=next(f for f in result['findings'] if f['rule_id']=='MEM-OUT-OF-BOUNDS' and f['method']=='bounded_execution')
        expected=next(i for i,line in enumerate(bad.splitlines(),1) if 'memref.load %x' in line)
        self.assertEqual(fault['locations'][0]['line'],expected)
        explicit=bad.replace('memref.load %x[%four] : memref<4xf32>',
            'memref.load %x[%four] : memref<4xf32> loc("original.py":999:2)')
        result=self.execute(self.group('debug_location',explicit),case,out_dir=self.root/'report')
        self.assertEqual(result['tool_status'],'mismatch')
        fault=next(f for f in result['findings'] if f['rule_id']=='MEM-OUT-OF-BOUNDS' and f['method']=='bounded_execution')
        self.assertIsNone(fault['locations'][0]['line'])
        self.assertIn('original.py',fault['details']['mlir_location']['ir_location'])
        self.assertIn('node_path',fault['details']['mlir_location'])

    def test_ssa_type_and_unknown_dialect_rejections(self):
        invalid=COPY.replace('arith.addf %a, %b','arith.addf %missing, %b')
        result=check(self.group('ssa',invalid,COPY),mlir_backend='official')
        self.assertEqual(result['tool_status'],'tool_error')
        runs=result['mlir_python_verification']['runs']
        self.assertEqual([r['status'] for r in runs],['invalid_ir','passed'])
        self.assertIn('undeclared SSA',runs[0]['diagnostic'])
        invalid=COPY.replace('arith.addf %a, %b : f32','arith.addi %a, %b : f32')
        self.assertEqual(check(self.group('type',invalid),mlir_backend='official')['tool_status'],'tool_error')
        unknown='module { "unavailable.side_effect"() : () -> () }'
        result=check(self.group('unknown_dialect',unknown),mlir_backend='official')
        self.assertEqual(result['tool_status'],'unsupported',result)
        self.assertTrue(all(r['failure_kind']=='dialect_incompatible' for r in result['mlir_python_verification']['runs']))

    def test_unknown_operation_and_fastmath_do_not_disappear(self):
        unknown=COPY.replace('%c = arith.constant 1.0 : f32',
            '%c = arith.constant 1.0 : f32\n    %false = arith.constant false\n    scf.if %false {\n      %atan = math.atan %c : f32\n    }')
        case=self.case('unknown')
        for name,text in (
            ('unknown_op',unknown),
            ('flags',COPY.replace('arith.addf %a, %b : f32','arith.addf %a, %b fastmath<reassoc> : f32')),
            ('module_attributes',COPY.replace('module {','module attributes {checker.target = "unmodeled"} {',1)),
        ):
            result=self.execute(self.group(name,text),case)
            self.assertEqual(result['mlir_python_verification']['status'],'passed',result)
            self.assertEqual(result['tool_status'],'unsupported',result['dynamic'])
            self.assertFalse(result['dynamic']['tests'])

    def test_parallel_conflict_is_detected(self):
        bad=COPY.replace('memref.store %v, %tmp[%i]','memref.store %v, %tmp[%zero]')
        result=self.execute(self.group('conflict',bad),self.case('conflict'))
        self.assertEqual(result['tool_status'],'mismatch')
        self.assertTrue(any(f['rule_id']=='PARALLEL-CONFLICT' for f in result['findings']))

    def test_layout_and_binding_fingerprint_are_enforced(self):
        layout=COPY.replace('%x: memref<4xf32>','%x: memref<4xf32, strided<[2]>>').replace(
            'memref.load %x[%i] : memref<4xf32>','memref.load %x[%i] : memref<4xf32, strided<[2]>>')
        result=self.execute(self.group('layout',layout),self.case('layout'))
        self.assertEqual(result['tool_status'],'unsupported',result)
        trace=self.group('version',COPY)
        manifest=json.loads((trace/'manifest.json').read_text());manifest['mlir_environment']={'extension_sha256':'different'}
        (trace/'manifest.json').write_text(json.dumps(manifest))
        result=check(trace,mlir_backend='official')
        self.assertEqual(result['tool_status'],'unsupported')
        self.assertTrue(all(r['failure_kind']=='version_incompatible' for r in result['mlir_python_verification']['runs']))

    @unittest.skipUnless(HAS_TVM,'requires both matching frontends in one environment')
    def test_real_where_and_sum_all_five_official_stages(self):
        for name in ('where','sum'):
            with self.subTest(name=name):
                result=check(FIXTURES/name,tir_backend='tvm',mlir_backend='official',dynamic=True,
                    case=FIXTURES/(name+'_case.py'),timeout=regression_timeout(180),out_dir=self.root/(name+'_report'))
                self.assertEqual(result['tool_status'],'ok',result['dynamic'])
                self.assertEqual(result['tir_verification']['status'],'passed')
                self.assertEqual(result['mlir_python_verification']['status'],'passed')
                self.assertEqual(len(result['dynamic']['tests']),15)
                self.assertTrue(all(t['status']=='match' for t in result['dynamic']['tests']))
                self.assertEqual({t['backend'] for t in result['dynamic']['tests']}, {'tvm_tir_nodes','mlir_operation_objects'})


if __name__=='__main__': unittest.main()

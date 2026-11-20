"""Optional official TVM acceptance, plus dependency-independent input guards.

Run the full module in the project's existing TileLang environment. A missing
TVM skips only real-official tests; it never becomes a successful verification.
"""
import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tests.support.timeouts import regression_timeout
from tir_mlir_checker import check
from tir_mlir_checker.trace_io import STAGES, InputError, load_trace

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT/'tests/fixtures/official_tir'
HAS_TVM = importlib.util.find_spec('tilelang') is not None


class OfficialInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copytree(FIXTURES/'where',self.root/'trace')
        self.trace = self.root/'trace'

    def test_object_tampering_is_input_error(self):
        file = self.trace/(STAGES[0]+'_object.json')
        file.write_bytes(file.read_bytes()+b' ')
        self.assertEqual(check(self.trace)['tool_status'],'input_error')

    def test_text_and_object_must_remain_bound(self):
        file = self.trace/(STAGES[0]+'.tir')
        file.write_text(file.read_text()+'\n')
        manifest = json.loads((self.trace/'manifest.json').read_text())
        record = manifest['stages'][0]
        raw = file.read_bytes()
        record.update(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),lines=len(raw.splitlines()))
        (self.trace/'manifest.json').write_text(json.dumps(manifest))
        self.assertEqual(check(self.trace)['tool_status'],'input_error')

    def test_object_cannot_escape_or_collide(self):
        manifest = json.loads((self.trace/'manifest.json').read_text())
        for name in ('../outside.json','manifest.json',STAGES[0]+'.tir'):
            manifest['tir_objects'][STAGES[0]]['file'] = name
            (self.trace/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(InputError): load_trace(self.trace)

    def test_bad_python_configuration_is_not_a_counterexample(self):
        result = check(self.trace,tir_backend='tvm',tir_python=str(self.root/'absent-python'))
        self.assertEqual(result['tool_status'],'unsupported')
        self.assertEqual(result['semantic_conclusion'],'not_established')
        self.assertEqual(result['capabilities']['formal_tir_verifier'],'not_completed')

    def test_invalid_backend_is_input_error(self):
        self.assertEqual(check(self.trace,tir_backend='guess')['tool_status'],'input_error')
        self.assertEqual(check(self.trace,tir_python='python')['tool_status'],'input_error')

    def test_node_location_is_not_presented_as_a_trace_line(self):
        from tir_mlir_checker.diagnostics import finding,agent_diagnostics,diagnostic_text
        item = finding('NUM-OUTPUT-DIFFERENCE','mismatch','bounded_execution','different',stage=STAGES[0],
            details={'tvm_location':{'node_path':'body.block.body','original_span':
                {'source':'operator.py','line':99,'column':4}}})
        diagnostics=agent_diagnostics({'findings':[item]})
        self.assertIsNone(diagnostics[0]['location']['line'])
        self.assertEqual(diagnostics[0]['location']['node_path'],'body.block.body')
        message=diagnostic_text(diagnostics)
        self.assertIn('operator.py:99:4 (not a Trace line)',message)
        self.assertIn('exported-text line unavailable',message)

    @unittest.skipIf(HAS_TVM,'checks missing optional dependency on the lightweight host')
    def test_missing_tvm_is_not_an_ast_fallback(self):
        result = check(self.trace,tir_backend='tvm')
        self.assertEqual(result['tool_status'],'unsupported')
        runs = result['tir_verification']['runs']
        self.assertEqual(len(runs),3)
        self.assertTrue(all(r['failure_kind']=='dependency_unavailable' for r in runs))
        self.assertFalse(any(f['formal_ir_verified'] for s,f in result['facts'].items() if s in STAGES[:3]))


@unittest.skipUnless(HAS_TVM,'requires the matching project TileLang/TVM environment')
class OfficialExecutionTests(unittest.TestCase):
    def setUp(self):
        from tilelang import tvm
        from tir_mlir_checker.tvm_frontend import identity
        self.tvm,self.identity = tvm,identity(tvm)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def assert_official(self,result):
        self.assertEqual(result['tir_verification']['status'],'passed',result)
        self.assertEqual(len(result['tir_verification']['runs']),3)
        for run in result['tir_verification']['runs']:
            self.assertEqual([c['status'] for c in run['checks']],['passed','passed'])

    def test_real_where_all_five_stages(self):
        result = check(FIXTURES/'where',tir_backend='tvm',dynamic=True,case=FIXTURES/'where_case.py',timeout=regression_timeout(120))
        self.assert_official(result)
        self.assertEqual(result['tool_status'],'ok',result['dynamic'])
        self.assertEqual(len(result['dynamic']['tests']),15)
        self.assertTrue(all(t['status']=='match' for t in result['dynamic']['tests']))
        nodes = [t for t in result['dynamic']['tests'] if t['stage'] in STAGES[:3]]
        self.assertTrue(all(t['backend']=='tvm_tir_nodes' for t in nodes))

    def test_real_sum_all_five_stages(self):
        result = check(FIXTURES/'sum',tir_backend='tvm',dynamic=True,case=FIXTURES/'sum_case.py',timeout=regression_timeout(120))
        self.assert_official(result)
        self.assertEqual(result['tool_status'],'ok',result['dynamic'])
        self.assertEqual(len(result['dynamic']['tests']),15)
        self.assertTrue(all(t['status']=='match' for t in result['dynamic']['tests']))

    def write_group(self,source,*,objects=True,name='trace'):
        from test_execution import write_group
        # Keep the unused MLIR side independent of the tested TIR programs.
        mlir = (FIXTURES/'where'/('03_mlir_linalg.mlir')).read_text()
        from tvm.script import ir as I, tir as T
        # from_source accepts a module body, not ordinary Python imports.
        obj = self.tvm.script.from_source('\n'.join(source.splitlines()[1:]),extra_vars={'I':I,'T':T})
        canonical = obj.script()
        trace = write_group(self.root/name,[canonical]*3+[mlir]*2)
        if objects:
            manifest = json.loads((trace/'manifest.json').read_text())
            manifest['tir_objects'] = {}
            for stage in STAGES[:3]:
                raw = self.tvm.ir.save_json(obj).encode();filename=stage+'_object.json'
                (trace/filename).write_bytes(raw)
                manifest['tir_objects'][stage] = {'file':filename,'sha256':hashlib.sha256(raw).hexdigest(),
                    'text_sha256':hashlib.sha256(canonical.encode()).hexdigest(),'tvm':self.identity}
            (trace/'manifest.json').write_text(json.dumps(manifest))
        return trace

    def run_tir(self,trace,case):
        result = check(trace,tir_backend='tvm',dynamic=True,case=case,stages=STAGES[:3],timeout=regression_timeout(120))
        self.assert_official(result)
        return result

    def test_minimal_add_copy_and_text_parser(self):
        source = '''from tvm.script import ir as I, tir as T
@I.ir_module
class Module:
    @T.prim_func
    def main(x: T.Buffer((4,), "float32"), y: T.Buffer((4,), "float32")):
        scratch = T.alloc_buffer((4,), "float32")
        for i in T.serial(4):
            scratch[i] = x[i] + T.float32(1)
        for i in T.parallel(4):
            y[i] = scratch[i]
'''
        case = self.root/'case.py'
        case.write_text('''import numpy as np
from tir_mlir_checker.cases import TestCase
def make_cases(seed):
    x=np.array([-3,0,2,99],dtype="float32")
    yield TestCase("distinct",(x,np.empty_like(x)),(1,),(x+np.float32(1),))
''')
        for objects in (True,False):
            trace = self.write_group(source,objects=objects,name='object' if objects else 'text')
            if not objects:
                # Harmless formatting/comment change must remain parsable.
                manifest = json.loads((trace/'manifest.json').read_text())
                for record in manifest['stages'][:3]:
                    file = trace/record['file'];raw = ('# harmless comment\n'+file.read_text()+'\n').encode()
                    file.write_bytes(raw)
                    record.update(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),lines=len(raw.splitlines()))
                (trace/'manifest.json').write_text(json.dumps(manifest))
            result = self.run_tir(trace,case)
            self.assertEqual(result['tool_status'],'ok',result['dynamic'])
            self.assertEqual(len(result['dynamic']['tests']),3)

    def test_four_semantic_mutations_with_independent_reference(self):
        source = '''from tvm.script import ir as I, tir as T
@I.ir_module
class Module:
    @T.prim_func
    def main(x: T.Buffer((4,4), "float32"), y: T.Buffer((4,), "float32")):
        acc = T.alloc_buffer((4,), "float32")
        for i in T.serial(4):
            acc[i] = T.float32(0)
            for j in T.serial(4):
                acc[i] = acc[i] + x[i,j]
            y[i] = acc[i]
'''
        case = self.root/'case.py'
        case.write_text('''import numpy as np
from tir_mlir_checker.cases import TestCase,Comparison
def make_cases(seed):
    x=np.arange(16,dtype="float32").reshape(4,4)+np.float32(.12345)
    expected=np.zeros(4,dtype="float32")
    for j in range(4): expected=np.add(expected,x[:,j],dtype=np.float32)
    yield TestCase("distinct",(x,np.empty(4,dtype="float32")),(1,),(expected,),comparison=Comparison("bitwise"))
''')
        self.assertEqual(self.run_tir(self.write_group(source,name='valid'),case)['tool_status'],'ok')
        changes = {
            'wrong_index':source.replace('x[i,j]','x[i,0]'),
            'missing_write':source.replace('y[i] = acc[i]','T.evaluate(0)'),
            'wrong_init':source.replace('T.float32(0)','T.float32(1)'),
            'early_precision':source.replace('acc[i] + x[i,j]','T.Cast("float32", T.Cast("bfloat16", acc[i] + x[i,j]))'),
            'missing_init':source.replace('acc[i] = T.float32(0)','T.evaluate(0)'),
        }
        for name,mutated in changes.items():
            with self.subTest(name=name):
                result = self.run_tir(self.write_group(mutated,name=name),case)
                self.assertEqual(result['tool_status'],'mismatch',result['dynamic'])
                self.assertEqual(result['semantic_conclusion'],'bounded_counterexample')
                self.assertTrue(any(f['rule_id'] in {'NUM-OUTPUT-DIFFERENCE','MEM-OUTPUT-UNWRITTEN','MEM-UNINITIALIZED'} for f in result['findings']))
                for f in result['findings']:
                    if f['method'] in {'bounded_execution','bounded_symbolic'}:
                        self.assertTrue(all(not loc.get('line') for loc in f['locations']))

    def test_version_mismatch_and_unknown_call_are_explicit(self):
        trace = self.root/'version';shutil.copytree(FIXTURES/'where',trace)
        manifest = json.loads((trace/'manifest.json').read_text())
        manifest['tir_objects'][STAGES[1]]['tvm']['git_commit'] = 'incompatible'
        (trace/'manifest.json').write_text(json.dumps(manifest))
        result = check(trace,tir_backend='tvm')
        self.assertEqual(result['tool_status'],'unsupported')
        runs = result['tir_verification']['runs']
        self.assertEqual([r['status'] for r in runs],['passed','unsupported','passed'])
        self.assertEqual(runs[1]['failure_kind'],'version_incompatible')
        source = '''from tvm.script import ir as I, tir as T
@I.ir_module
class Module:
    @T.prim_func
    def main(x: T.Buffer((4,), "float32"), y: T.Buffer((4,), "float32")):
        for i in T.serial(4):
            y[i] = T.call_extern("float32", "unimplemented", x[i])
'''
        case = self.root/'unknown_case.py'
        case.write_text('''import numpy as np
from tir_mlir_checker.cases import TestCase
def make_cases(seed):
    yield TestCase("unknown",(np.ones(4,dtype="float32"),np.empty(4,dtype="float32")),(1,),(np.ones(4,dtype="float32"),))
''')
        result = self.run_tir(self.write_group(source,name='unknown'),case)
        self.assertEqual(result['tool_status'],'unsupported')
        records = result['dynamic']['tests'] + list(result['dynamic']['stage_support'].values())
        self.assertTrue(any('tir.call_extern' in t.get('reason','') for t in records))

    def test_official_ssa_rejection_is_not_a_semantic_counterexample(self):
        tvm = self.tvm
        var = tvm.tir.Var('duplicate','int32')
        body = tvm.tir.SeqStmt([
            tvm.tir.LetStmt(var,tvm.tir.IntImm('int32',1),tvm.tir.Evaluate(var)),
            tvm.tir.LetStmt(var,tvm.tir.IntImm('int32',2),tvm.tir.Evaluate(var))])
        obj = tvm.IRModule({'main':tvm.tir.PrimFunc([],body)})
        self.assertFalse(tvm.tir.analysis.verify_ssa(obj['main']))
        from test_execution import write_group
        mlir = (FIXTURES/'where'/'03_mlir_linalg.mlir').read_text()
        text = obj.script();trace = write_group(self.root/'invalid',[text]*3+[mlir]*2)
        manifest = json.loads((trace/'manifest.json').read_text())
        manifest['tir_objects'] = {}
        for stage in STAGES[:3]:
            raw = tvm.ir.save_json(obj).encode();filename=stage+'_object.json'
            (trace/filename).write_bytes(raw)
            manifest['tir_objects'][stage] = {'file':filename,'sha256':hashlib.sha256(raw).hexdigest(),
                'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'tvm':self.identity}
        (trace/'manifest.json').write_text(json.dumps(manifest))
        result = check(trace,tir_backend='tvm')
        self.assertEqual(result['tool_status'],'tool_error')
        self.assertEqual(result['tir_verification']['status'],'error')
        self.assertTrue(all(r['checks'][0]['status']=='invalid_ir' for r in result['tir_verification']['runs']))
        self.assertEqual(result['semantic_conclusion'],'not_established')


if __name__ == '__main__':
    unittest.main()

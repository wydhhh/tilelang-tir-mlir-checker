"""Agent translations are supplementary executable models, never IR verdicts."""
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from tir_mlir_checker import check
from tir_mlir_checker.trace_io import STAGES
from test_execution import write_group, np


@unittest.skipIf(np is None, 'NumPy required')
class TranslationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        tir = '@T.prim_func\ndef main(x: T.handle, y: T.handle):\n    T.unknown()\n'
        mlir = 'module {\n  func.func @main() {\n    custom.unknown\n    return\n  }\n}\n'
        self.trace = write_group(self.root/'trace', [tir]*3 + [mlir]*2)
        self.case = self.root/'case.py'
        self.case.write_text('''import numpy as np
from tir_mlir_checker.cases import TestCase
def make_cases(seed):
    x = np.array([-2, 0, 3], dtype=np.float32)
    yield TestCase('sample', (x, np.empty_like(x)), (1,), (x * np.float32(2),))
''')
        self.translation = self.root/'generated.py'

    def generated(self, body='return {1: arguments[0] * np.float32(2)}', **changes):
        import hashlib
        spec = {'sha256': hashlib.sha256((self.trace/'03_mlir_linalg.mlir').read_bytes()).hexdigest(),
                'entry': 'run', 'assumptions': ['Synthetic model; no physical parallelism'],
                'line_map': [{'ir': [2, 4], 'python': [2, 3]}]}
        spec.update(changes)
        self.translation.write_text('import numpy as np\ndef run(arguments, symbols):\n    '+body+
                                    '\nTRANSLATIONS = '+repr({STAGES[3]: spec})+'\n')

    def run_translation(self, **kwargs):
        return check(self.trace, case=self.case, dynamic=True, translation=self.translation,
                     timeout=kwargs.pop('timeout', 10), **kwargs)

    def test_supplement_pass_does_not_upgrade_unsupported_and_has_replay_evidence(self):
        self.generated()
        r = self.run_translation(out_dir=self.root/'out')
        self.assertEqual(r['tool_status'], 'unsupported')
        self.assertEqual(r['semantic_conclusion'], 'not_established')
        a = r['agent_translation']
        self.assertEqual((a['status'], a['input_binding']), ('passed', 'identical'))
        self.assertEqual(a['fidelity'], 'unverified_agent_translation')
        self.assertEqual(a['tests'][0]['bitwise_equal_elements'], 3)
        with zipfile.ZipFile(r['artifacts']['evidence']) as z:
            self.assertEqual(z.read('agent_translation/translation.py'), self.translation.read_bytes())
            arrays = np.load(io.BytesIO(z.read('agent_translation/arrays.npz')), allow_pickle=False)
            self.assertIn('case0_expected1', arrays)
        self.assertIn('Agent 翻译后样本验证', Path(r['artifacts']['report']).read_text())

    def test_wrong_generated_code_has_counterexample_but_not_compiler_error(self):
        self.generated('return {1: arguments[0] * np.float32(3)}')
        r = self.run_translation()
        self.assertEqual(r['tool_status'], 'unsupported')
        a = r['agent_translation']
        self.assertEqual(a['status'], 'mismatch')
        self.assertEqual(a['tests'][0]['first_difference']['index'], [0])
        f = next(f for f in r['findings'] if f['rule_id']=='AGENT-TRANSLATION-DIFFERENCE')
        self.assertEqual(f['status'], 'uncertain')

    def test_stale_ir_hash_is_not_executed(self):
        self.generated('raise AssertionError("must not run")', sha256='0'*64)
        r = self.run_translation()
        self.assertEqual(r['agent_translation']['status'], 'unsupported')
        self.assertIn('哈希', r['agent_translation']['reason'])

    def test_invalid_mapping_is_rejected(self):
        self.generated(line_map=[{'ir': [1, 999], 'python': [2, 3]}])
        r = self.run_translation()
        self.assertEqual(r['agent_translation']['status'], 'unsupported')
        self.assertIn('映射', r['agent_translation']['reason'])

    def test_wrong_shape_and_missing_outputs_are_not_passed(self):
        for code in ['return {}', 'return {1: arguments[0][:1]}',
                     'return {1: arguments[0].astype(np.float64)}']:
            self.generated(code)
            self.assertNotEqual(self.run_translation()['agent_translation']['status'], 'passed')

    def test_translation_timeout_preserves_sources_and_original_status(self):
        self.generated('exec("while True: pass")')
        r = self.run_translation(timeout=1)
        self.assertEqual(r['tool_status'], 'unsupported')
        self.assertEqual(r['agent_translation']['failure_kind'], 'timeout')
        self.assertIn('agent_translation/translation.py', r['_evidence'])

    def test_nondeterministic_cases_do_not_claim_same_input(self):
        self.generated()
        self.case.write_text(self.case.read_text().replace(
            'x = np.array([-2, 0, 3], dtype=np.float32)',
            'x = np.random.default_rng().normal(size=3).astype(np.float32)'))
        r = self.run_translation()
        self.assertEqual(r['agent_translation']['status'], 'error')
        self.assertEqual(r['agent_translation']['input_binding'], 'different')

    def test_translation_requires_explicit_case_and_dynamic(self):
        self.generated()
        r = check(self.trace, translation=self.translation)
        self.assertEqual(r['tool_status'], 'input_error')

    def test_original_mismatch_cannot_be_hidden_by_supplement(self):
        # Use real Where IR with reversed condition and a correctly generated reference model.
        from test_execution import WHERE, CASE
        sources = [(WHERE/(s+('.tir' if s in STAGES[:3] else '.mlir'))).read_text() for s in STAGES]
        sources[3] = sources[3].replace('arith.cmpi ne, %23', 'arith.cmpi eq, %23')
        self.trace = write_group(self.root/'wrong_where', sources)
        self.generated('return {3: np.where(arguments[0] != 0, arguments[1], arguments[2])}')
        # This test deliberately declares an unfaithful translation: its match must not mask the IR error.
        r = check(self.trace, case=CASE, dynamic=True, translation=self.translation, timeout=60)
        self.assertEqual(r['tool_status'], 'mismatch')
        self.assertEqual(r['semantic_conclusion'], 'bounded_counterexample')
        self.assertEqual(r['agent_translation']['status'], 'passed')


if __name__ == '__main__':
    unittest.main()

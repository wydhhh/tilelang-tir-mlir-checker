"""Human report contracts; structured evidence and raw diagnostics stay lossless."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from tir_mlir_checker.diagnostics import agent_diagnostics, diagnostic_text, finding
from tir_mlir_checker.report import render_report, write_report
from tir_mlir_checker.trace_io import STAGES


def sample_result():
    cases = [{'name': name, 'outputs': [3], 'arguments': [
        {'dtype': dtype, 'shape': [35]} for dtype in ['uint8', 'float32', 'float32', 'float32']]}
        for name in ['all_zero', 'all_one', 'nonzero', 'alternating', 'random', 'boundary',
                     'ieee_alternating', 'ieee_zero', 'ieee_one']]
    tests = [dict(stage=stage, case=case['name'], output=3, status='match', elements=35,
                  max_finite_abs_error=0.0, policy={'mode': 'bitwise'})
             for case in cases for stage in STAGES]
    return {'tool_status': 'ok', 'tool_version': '0.12.0', 'semantic_conclusion': 'bounded_checks_passed',
            'input': {'root': '/not-an-output/trace', 'run_id': 'sample-run'},
            'operator': {'name': 'Where', 'ir_functions': {s: ['main'] for s in STAGES}},
            'summary': {'match': 20, 'uncertain': 4, 'mismatch': 0},
            'checks': [{'id': r, 'status': 'uncertain', 'scope': scope}
                       for r, scope in [('P1-formula','公式'), ('P2-dataflow','数据流'),
                                        ('P3-precision','精度'), ('P4-control','控制流')]],
            'dynamic': {'status': 'passed', 'complete': True, 'executed': True, 'requested_dynamic': True,
                        'tests': tests, 'cases': cases,
                        'stage_support': {s: {'backend': 'tvm_tir_nodes' if i < 3 else 'mlir_operation_objects'}
                                          for i, s in enumerate(STAGES)}},
            'tir_verification': {'status': 'passed', 'runs': []},
            'mlir_python_verification': {'status': 'passed', 'runs': []},
            'mlir_verification': {'status': 'passed', 'runs': []},
            'findings': [], 'diagnostics': [], 'facts': {'retained': 'unchanged'}}


def fail_result():
    r = sample_result()
    r.update(tool_status='mismatch', semantic_conclusion='bounded_counterexample')
    r['summary'] = {'match': 17, 'uncertain': 6, 'mismatch': 1}
    r['dynamic']['status'] = 'mismatch'
    for test in r['dynamic']['tests']:
        if test['stage'] == STAGES[3]:
            test.update(status='mismatch', max_finite_abs_error=4.0)
            r['findings'].append(finding('NUM-OUTPUT-DIFFERENCE', 'mismatch', 'bounded_execution',
                '实际输出与独立参考不一致', locations=[
                    {'stage': STAGES[3], 'file': '/example/03_mlir_linalg.mlir', 'line': 226,
                     'excerpt': 'memref.store %24, %arg3[%13] : memref<35xf32>'},
                    # Secondary locations vary by case; the primary error is still one group.
                    {'stage': STAGES[3], 'line': 167 + len(r['findings'])}],
                details={'case': test['case'], 'output': 3,
                         'first_difference': {'index': [0], 'actual': {'value': '1.0', 'bits': '0000803f'},
                                              'expected': {'value': '-3.0', 'bits': '000040c0'}}}))
    return r


class ReportTests(unittest.TestCase):
    def test_passing_report_has_three_sections_deduplicated_signatures_and_reviews(self):
        r = sample_result()
        original = copy.deepcopy(r)
        text = render_report(r)
        self.assertEqual([line for line in text.splitlines() if line.startswith('## ')], [
            '## 1. 检查结论', '## 2. 怎么查的与测试范围', '## 3. 问题定位与 Agent 复核'])
        self.assertIn('✅ **通过**', text)
        self.assertIn('45 / 45 次逐位一致', text)
        self.assertIn('9 组输入 × 5 阶段', text)
        self.assertIn('累计 1575 个输出元素', text)
        self.assertEqual(text.count('uint8[35]'), 1)
        self.assertEqual(text.count('float32[35]'), 3)
        self.assertNotIn("'match':", text)
        self.assertNotIn('20 match', text)
        self.assertIn('**自动报错诊断：** 无。', text)
        self.assertEqual(text.count('[待 Agent 复核]'), 4)
        self.assertIn('TIR 00–02 读取 / SSA / 结构：通过', text)
        self.assertIn('MLIR 03–04 解析 / verifier：通过', text)
        self.assertIn('mlir-opt 03–04：通过', text)
        self.assertEqual(len([line for line in text.splitlines() if line[:2] in ('1.', '2.', '3.')]), 3)
        self.assertEqual(r, original)

    def test_nine_failures_fold_into_one_card_with_first_witness_and_root_slot(self):
        r = fail_result()
        text = render_report(r)
        self.assertIn('❌ **失败**', text)
        self.assertIn('共 9 次模拟失败', text)
        self.assertIn('36 / 45 次逐位一致', text)
        self.assertEqual(text.count('[NUM-OUTPUT-DIFFERENCE]'), 1)
        self.assertIn('影响用例（9 组）', text)
        self.assertIn('03_mlir_linalg.mlir:226', text)
        self.assertIn('memref.store %24', text)
        self.assertIn('`all_zero` 的 `arg3(out)[0]`：实际 `1.0`，独立参考 `-3.0`', text)
        self.assertIn('> **Agent 根因分析：** [待 Agent 沿 IR 数据流回溯确认]', text)
        self.assertIn('控制流', text)

    def test_multi_output_comparisons_are_not_counted_as_extra_executions(self):
        r = sample_result()
        extra = copy.deepcopy(r['dynamic']['tests'])
        for test in extra:
            test['output'] = 4
        r['dynamic']['tests'] += extra
        text = render_report(r)
        self.assertIn('45 / 45 次逐位一致', text)
        self.assertIn('累计 3150 个输出元素', text)
        self.assertNotIn('90 次模拟', text)

    def test_uncertain_check_and_located_finding_are_one_review_item(self):
        r = sample_result()
        r['checks'].append({'id': 'S-005', 'status': 'uncertain', 'scope': '表达式差异',
                            'evidence': [{'stage': STAGES[2], 'lines': [30]}]})
        r['findings'] = [finding('FLOW-EXPRESSION-DIFFERENCE', 'uncertain', 'bounded_symbolic',
                                '表达式差异', stage=STAGES[2], line=30)] * 3
        text = render_report(r)
        self.assertIn('待 Agent 复核项（5 项）', text)
        self.assertEqual(text.count('FLOW-EXPRESSION-DIFFERENCE'), 1)
        self.assertNotIn('S-005', text)

    def test_varied_shapes_are_grouped_not_hidden(self):
        r = sample_result()
        r['dynamic']['cases'][-1]['arguments'][0]['shape'] = [17]
        text = render_report(r)
        self.assertEqual(text.count('uint8[35]'), 1)
        self.assertEqual(text.count('uint8[17]'), 1)
        self.assertIn('ieee_one', text)

    def test_static_and_unsupported_checks_never_claim_dynamic_success(self):
        r = sample_result()
        r['dynamic'] = {'status': 'not_requested', 'executed': False}
        r.pop('mlir_verification')
        self.assertIn('静态检查完成', render_report(r))
        self.assertNotIn('✅', render_report(r))
        r['tool_status'] = 'unsupported'
        r['tir_verification']['status'] = 'unsupported'
        self.assertIn('检查未完成', render_report(r))
        self.assertIn('依赖或格式不支持', render_report(r))
        self.assertNotIn('✅', render_report(r))

    def test_invalid_official_ir_is_failure_even_without_dynamic_tests(self):
        r = sample_result()
        r['tool_status'] = 'tool_error'
        r['dynamic'] = {'status': 'not_requested', 'executed': False}
        r['tir_verification'] = {'status': 'error', 'runs': [{'stage': STAGES[0], 'status': 'invalid_ir'}]}
        self.assertIn('❌ **失败**', render_report(r))
        self.assertIn('00_tir_initial', render_report(r))

    def test_tolerance_and_translation_are_explicit_and_separate(self):
        r = sample_result()
        for test in r['dynamic']['tests']:
            test['policy'] = {'mode': 'tolerance', 'rtol': 1e-5, 'atol': 1e-6}
        r['agent_translation'] = {'status': 'passed', 'input_binding': 'identical',
                                  'tests': [dict(stage=STAGES[3],case='all_zero',status='match')]}
        text = render_report(r)
        self.assertIn('45 / 45 次容差一致', text)
        self.assertIn('rtol=1e-05, atol=1e-06', text)
        self.assertIn('Agent 翻译后样本验证（单列）', text)
        self.assertIn('输入绑定：identical', text)

    def test_evidence_retains_every_finding_and_original_rule_dictionary(self):
        r = fail_result()
        original = copy.deepcopy(r)
        expected_diagnostics = diagnostic_text(agent_diagnostics(r))
        with tempfile.TemporaryDirectory() as temp:
            write_report(r, Path(temp)/'result')
            with zipfile.ZipFile(r['artifacts']['evidence']) as z:
                saved = json.loads(z.read('result.json'))
                self.assertEqual(saved['summary'], original['summary'])
                self.assertEqual(saved['findings'], original['findings'])
                self.assertEqual(saved['dynamic'], original['dynamic'])
                self.assertEqual(json.loads(z.read('facts.json')), original['facts'])
                self.assertEqual(z.read('diagnostics.txt').decode(), expected_diagnostics)
                self.assertEqual(z.read('diagnostics.txt').decode().count('error:'), 9)

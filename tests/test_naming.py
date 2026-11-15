"""Artifact identity, safe names, and the real CLI run layout."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from tir_mlir_checker import check
from tir_mlir_checker.cli import main
from tir_mlir_checker.naming import identify_operator, validate_operator_name
from tir_mlir_checker.trace_io import InputError

TRACE = Path(__file__).resolve().parent / 'fixtures/traces/十算子-2026-11-11/Where'


class NamingTests(unittest.TestCase):
    def test_explicit_name_preserves_ir_identity_and_packaged_artifact_links(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'Where'/'20261113_run01'
            r = check(TRACE, out_dir=out, operator_name='选择算子')
            self.assertEqual(r['operator']['name'], '选择算子')
            self.assertEqual(r['operator']['name_source'], 'explicit')
            self.assertTrue(any(r['operator']['ir_functions'].values()))
            self.assertEqual({p.name for p in out.iterdir()},
                             {'选择算子 自动检查记录.md', '选择算子 检查证据.zip'})
            report = Path(r['artifacts']['report']).read_text()
            self.assertEqual(report.splitlines()[0], '# 选择算子 自动检查记录')
            self.assertIn('(<选择算子 检查证据.zip>)', report)
            self.assertEqual(len([s for s in report.splitlines() if s.startswith('## ')]), 3)
            self.assertFalse((out/r['report_naming']['final_report']).exists())
            with zipfile.ZipFile(r['artifacts']['evidence']) as z:
                saved = json.loads(z.read('result.json'))
                self.assertEqual(saved['artifacts'], r['artifacts'])
                self.assertEqual(saved['operator'], r['operator'])
                self.assertEqual(saved['findings'], r['findings'])
            with self.assertRaises(InputError):
                check(TRACE, out_dir=out, operator_name='选择算子')

    def test_identity_fallbacks_do_not_strip_ir_names(self):
        r = {'input': {'root': '/tmp/fallback', 'metadata': {'name': 'from_manifest'}},
             'facts': {'02': {'functions': [{'name': 'actual_kernel'}]}}}
        self.assertEqual(identify_operator(r)['name'], 'from_manifest')
        del r['input']['metadata']['name']
        self.assertEqual(identify_operator(r)['name'], 'actual_kernel')
        r['facts'] = {}
        self.assertEqual(identify_operator(r)['name'], 'fallback')

    def test_unsafe_explicit_names_are_rejected_before_writes(self):
        for name in ('../other', '/tmp/x', '..', '.', '', 'a/b', 'a\\b', 'bad\nname', 'x'*101):
            with self.subTest(name=name), self.assertRaises(InputError):
                validate_operator_name(name)

    def test_run_writes_operator_artifacts_next_to_trace_and_refuses_repeat(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp).resolve()/'Where'/'run01'
            def fake_export(adapter, dest, timeout):
                shutil.copytree(TRACE, dest)
                return {'tool_status': 'ok', 'exit_code': 0, 'ir': []}
            # Compilation is mocked; check, report writing and CLI JSON are real.
            with patch('tir_mlir_checker.capture.export_trace', side_effect=fake_export), \
                    contextlib.redirect_stdout(io.StringIO()) as stream:
                code = main(['run', '--adapter', 'unused.py', '--out', str(out),
                             '--operator', 'Where', '--static-only'])
            self.assertEqual(code, 0)
            response = json.loads(stream.getvalue())
            self.assertEqual(Path(response['artifacts']['report']).parent, out)
            self.assertEqual({p.name for p in out.iterdir()},
                             {'trace', 'Where 自动检查记录.md', 'Where 检查证据.zip'})
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['run', '--adapter', 'unused.py', '--out', str(out)]), 2)

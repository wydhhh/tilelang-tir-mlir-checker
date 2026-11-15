"""Delivery contracts: capture failures, precise diagnostics, and CLI boundaries."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tir_mlir_checker.capture import export_trace
from tir_mlir_checker.cli import main
from tir_mlir_checker.diagnostics import agent_diagnostics, diagnostic_text
from tir_mlir_checker.trace_io import InputError


class DeliveryTests(unittest.TestCase):
    def test_diagnostics_preserve_status_and_secondary_location(self):
        data = {"findings": [{"rule_id": "MEM-OUT-OF-BOUNDS", "status": "mismatch",
                             "method": "bounded_execution", "message": "index 3 outside [3,4]",
                             "locations": [{"file": "02.tir", "line": 26, "excerpt": "x[t,i]"},
                                           {"file": "02.tir", "line": 25}],
                             "details": {"index": [3, 0]}}]}
        items = agent_diagnostics(data)
        self.assertEqual(items[0]["status"], "mismatch")
        self.assertEqual(items[0]["details"]["index"], [3, 0])
        self.assertIn("02.tir:26: error:", diagnostic_text(items))
        self.assertIn("02.tir:25: note:", diagnostic_text(items))

    def test_offline_failure_is_not_success_and_has_logs(self):
        import platform
        if platform.machine() == "riscv64":
            self.skipTest("requires non-RISC-V host")
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            (p / "adapter.py").write_text("def make_kernel(): raise AssertionError('must not execute')")
            result = export_trace(p / "adapter.py", p / "trace")
            self.assertEqual(result["tool_status"], "unsupported")
            self.assertEqual(result["exit_code"], 4)
            self.assertTrue((p / "trace/采集证据.zip").exists())
            self.assertFalse((p / "trace/manifest.json").exists())

    def test_export_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp); a = p / "a.py"; a.write_text("")
            with self.assertRaises(InputError):
                export_trace(a, p)
            self.assertEqual(a.read_text(), "")

    def test_removed_cli_modes_are_rejected(self):
        for command in ['batch','benchmark','example','collect','skill']:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                main([command])
            self.assertEqual(exc.exception.code,2)

    def test_run_preserves_real_export_and_checks_same_trace(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp).resolve()/'run'
            captured={'exit_code':0,'tool_status':'ok','ir':[]}
            with patch('tir_mlir_checker.capture.export_trace',return_value=captured) as export, \
                 patch('tir_mlir_checker.cli.check',return_value={'exit_code':0}) as check, \
                 patch('tir_mlir_checker.cli._response',return_value={'tool_status':'ok'}), \
                 contextlib.redirect_stdout(io.StringIO()) as stdout:
                code=main(['run','--adapter','operator.py','--out',str(out),'--static-only'])
            self.assertEqual(code,0)
            export.assert_called_once_with('operator.py',out/'trace',timeout=600)
            self.assertEqual(check.call_args.args,(out/'trace',))
            self.assertEqual(check.call_args.kwargs['out_dir'],out)
            self.assertFalse(check.call_args.kwargs['dynamic'])
            self.assertIsNone(check.call_args.kwargs['case'])
            self.assertEqual(json.loads(stdout.getvalue())['capture'],captured)

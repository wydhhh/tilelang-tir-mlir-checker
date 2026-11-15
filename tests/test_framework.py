"""Framework contracts and failure modes; not semantic-equivalence tests."""
import contextlib
import hashlib
import io
import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired
from unittest.mock import patch

from tir_mlir_checker import check
from tir_mlir_checker.cli import main
from tir_mlir_checker.trace_io import InputError, STAGES, excerpt

PROJECT = Path(__file__).resolve().parent / "fixtures"
BATCH = PROJECT / "traces/十算子-2026-11-11"


class FrameworkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="tir-mlir-framework-test-")
        self.addCleanup(self.tmp.cleanup)
        self.temp = Path(self.tmp.name)

    def copy_group(self):
        target = self.temp / "pre_apply_mix"
        shutil.copytree(PROJECT / "traces/pre_apply_mix", target,
                        ignore=lambda parent, names: [name for name in names if name.endswith(".zip")])
        return target

    def change_manifest(self, target, mutate):
        path = target / "manifest.json"
        value = json.loads(path.read_text())
        mutate(value)
        path.write_text(json.dumps(value))

    def test_every_real_group_and_legacy_layout(self):
        groups = sorted(p for p in BATCH.iterdir() if p.is_dir())
        self.assertEqual(len(groups), 11)
        groups += [PROJECT / "traces/pre_apply_mix", PROJECT / "traces/post"]
        for group in groups:
            with self.subTest(group=group):
                result = check(group)
                self.assertEqual(result["tool_status"], "ok")
                self.assertEqual(result["semantic_conclusion"], "not_established")
                self.assertEqual(len(result["input"]["stages"]), 5)
                self.assertTrue(all(set(s["verified_fields"]) == {"sha256", "bytes", "lines"}
                                    for s in result["input"]["stages"]))
                self.assertEqual(set(result["facts"]), set(STAGES))
                self.assertEqual(result["dynamic"]["status"], "not_requested")
                self.assertFalse(result["historical_execution"]["executed_in_this_check"])

    def test_tamper_is_input_error_not_semantic_mismatch(self):
        target = self.copy_group()
        path = target / "02_tir_optimized.tvmscript"
        path.write_bytes(path.read_bytes() + b"\n# changed\n")
        result = check(target)
        self.assertEqual(result["exit_code"], 2)
        self.assertEqual(result["facts"], {})
        self.assertEqual(result["summary"]["mismatch"], 0)

    def test_missing_stage(self):
        target = self.copy_group()
        (target / "03_mlir_linalg.mlir").unlink()
        self.assertEqual(check(target)["tool_status"], "input_error")

    def test_duplicate_stage(self):
        target = self.copy_group()
        self.change_manifest(target, lambda m: m["stages"].append(dict(m["stages"][0])))
        self.assertEqual(check(target)["tool_status"], "input_error")

    def test_cannot_read_another_groups_file(self):
        target = self.copy_group()
        self.change_manifest(target, lambda m: m["stages"][0].update(file="../other/00.tir"))
        result = check(target)
        self.assertEqual(result["tool_status"], "input_error")
        self.assertIn("越出", result["diagnostics"][0])

    def test_broken_batch_zip_is_not_silently_ignored(self):
        target = self.temp / "Where"
        target.mkdir()
        (self.temp / "验证与来源.zip").write_bytes(b"not a zip")
        self.assertEqual(check(target)["tool_status"], "input_error")

    def test_unknown_batch_group_is_not_guessed(self):
        target = self.temp / "unknown"
        target.mkdir()
        with zipfile.ZipFile(self.temp / "验证与来源.zip", "w") as archive:
            archive.writestr("manifest.json", json.dumps({"schema_version": 1, "groups": []}))
        self.assertEqual(check(target)["tool_status"], "input_error")

    def test_missing_manifest_is_unverified(self):
        target = self.copy_group()
        (target / "manifest.json").unlink()
        result = check(target)
        self.assertEqual(result["tool_status"], "ok")
        item = next(c for c in result["checks"] if c["id"] == "I1-integrity")
        self.assertEqual(item["status"], "uncertain")

    def test_unknown_schema_is_input_error(self):
        target = self.copy_group()
        self.change_manifest(target, lambda m: m.update(schema_version=2))
        self.assertEqual(check(target)["tool_status"], "input_error")

    def test_dynamic_request_never_counts_history_as_execution(self):
        case = self.temp / "case.py"
        case.write_text("# No make_cases: not a valid execution adapter\n")
        result = check(BATCH / "Where", dynamic=True, case=case)
        self.assertEqual(result["exit_code"], 4)
        self.assertEqual(result["dynamic"]["status"], "unsupported")
        self.assertFalse(result["dynamic"]["executed"])

    def test_case_without_dynamic_is_not_ignored(self):
        result = check(BATCH / "Where", case="case.py")
        self.assertEqual(result["tool_status"], "input_error")

    def test_requested_verifier_missing_is_unsupported(self):
        result = check(BATCH / "Where", mlir_opt=str(self.temp / "missing-mlir-opt"))
        self.assertEqual(result["exit_code"], 4)
        self.assertEqual(result["capabilities"]["formal_mlir_verifier"], "unsupported")

    def test_verifier_diagnostic_is_not_a_semantic_mismatch(self):
        with patch("tir_mlir_checker.analysis.shutil.which", return_value="/test/mlir-opt"), \
             patch("tir_mlir_checker.analysis.subprocess.run", return_value=CompletedProcess(
                 args=[], returncode=1, stdout="", stderr="unknown dialect or invalid IR")):
            result = check(BATCH / "Where", mlir_opt="/test/mlir-opt")
        self.assertEqual(result["exit_code"], 3)
        self.assertEqual(result["summary"]["mismatch"], 0)
        self.assertEqual(result["semantic_conclusion"], "not_established")

    def test_verifier_timeout_returns_tool_error(self):
        with patch("tir_mlir_checker.analysis.shutil.which", return_value="/test/mlir-opt"), \
             patch("tir_mlir_checker.analysis.subprocess.run", side_effect=TimeoutExpired("mlir-opt", 30)):
            result = check(BATCH / "Where", mlir_opt="/test/mlir-opt")
        self.assertEqual(result["exit_code"], 3)
        self.assertEqual(result["capabilities"]["formal_mlir_verifier"], "error")

    def test_tir_source_is_not_executed(self):
        target = self.copy_group()
        path = target / "00_tir_initial.tvmscript"
        path.write_text("raise RuntimeError('TVMScript is data')\n" + path.read_text())
        raw = path.read_bytes()
        self.change_manifest(target, lambda m: m["stages"][0].update(
            sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), lines=len(raw.splitlines())))
        result = check(target)
        self.assertEqual(result["tool_status"], "ok")
        self.assertFalse(result["facts"][STAGES[0]]["formal_ir_verified"])

    def test_source_queries_keep_original_line_numbers(self):
        group = BATCH / "Where"
        result = excerpt(group, "03", 62, 68)
        actual = (group / "03_mlir_linalg.mlir").read_text().splitlines()
        self.assertEqual(result["lines"][0], {"line": 62, "text": actual[61]})
        self.assertEqual(result["lines"][-1], {"line": 68, "text": actual[67]})
        self.assertEqual(result["sha256"], hashlib.sha256((group / "03_mlir_linalg.mlir").read_bytes()).hexdigest())

    def test_source_query_rejects_invalid_ranges(self):
        for stage, start, end in [("03", 0, 3), ("03", 1, 401), ("03", 99999, 99999), ("05", 1, 3)]:
            with self.subTest(stage=stage, start=start, end=end), self.assertRaises(InputError):
                excerpt(BATCH / "Where", stage, start, end)

    def test_reports_are_two_files_and_machine_payload_is_recoverable(self):
        out = self.temp / "result"
        result = check(BATCH / "Where", out_dir=out)
        self.assertEqual({p.name for p in out.iterdir()}, {Path(result["artifacts"]["report"]).name, Path(result["artifacts"]["evidence"]).name})
        with zipfile.ZipFile(result["artifacts"]["evidence"]) as archive:
            self.assertIsNone(archive.testzip())
            saved = json.loads(archive.read("result.json"))
            facts = json.loads(archive.read(saved["facts_member"]))
        self.assertEqual(saved["summary"], result["summary"])
        self.assertEqual(facts, result["facts"])
        self.assertIn("## 3. 问题与 Agent 判断", Path(result["artifacts"]["report"]).read_text())

    def test_reports_do_not_overwrite_or_write_into_trace(self):
        target = self.copy_group()
        for out in [target / "report", self.temp]:
            with self.subTest(out=out), self.assertRaises(InputError):
                check(target, out_dir=out)
        self.assertFalse((target / "report").exists())

    def test_formula_mutation_with_fresh_hash_remains_unproved(self):
        target = self.copy_group()
        path = target / "02_tir_optimized.tvmscript"
        source = path.read_text()
        self.assertIn("x[", source)
        # A real semantic change after refreshing integrity metadata must not be
        # described as verified semantics by this framework's local observations.
        source = source.replace("x[", "x[0 * ", 1)
        path.write_text(source)
        raw = path.read_bytes()
        self.change_manifest(target, lambda m: m["stages"][2].update(
            sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), lines=len(source.splitlines())))
        result = check(target)
        self.assertEqual(result["tool_status"], "ok")
        self.assertEqual(result["semantic_conclusion"], "not_established")
        formula = next(c for c in result["checks"] if c["id"] == "P1-formula")
        self.assertEqual(formula["status"], "uncertain")

    def test_cli_emits_single_json_response_and_consistent_exit(self):
        out = self.temp / "cli"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rc = main(["check", "--trace", str(BATCH / "Where"), "--out", str(out), "--dynamic"])
        response = json.loads(stdout.getvalue())
        self.assertEqual(rc, response["exit_code"])
        self.assertEqual(rc, 4)
        self.assertTrue(Path(response["artifacts"]["evidence"]).is_file())


if __name__ == "__main__":
    unittest.main()

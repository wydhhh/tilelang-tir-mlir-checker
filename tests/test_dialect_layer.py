"""Dialect-layer contract checks: facts, profiles, diagnostics, backends, CLI."""
import json
from pathlib import Path

import pytest

from tir_mlir_checker.dialect import (analyze_module, check_module, check_trace,
                                      load_profile, read_module, template_from_module)
from tir_mlir_checker.dialect.model import DialectReadError
from tir_mlir_checker.dialect.profile import profile_from_dict

HERE = Path(__file__).resolve().parent
TOOL = HERE.parent
WHERE = TOOL / "tests/fixtures/traces/十算子-2026-11-11/Where"
STAGE03 = WHERE / "03_mlir_linalg.mlir"

AME_MODULE = """module {
  func.func @main(%a: memref<64x128xf16>, %b: memref<128x64xf16>,
                  %c: memref<64x64xf16>) {
    "ame.mma"(%a, %b, %c) : (memref<64x128xf16>, memref<128x64xf16>, memref<64x64xf16>) -> ()
    return
  }
}
"""

AME_F32_MODULE = AME_MODULE.replace("64x64xf16", "64x64xf32")
AME_UNKNOWN_OP = AME_MODULE.replace('"ame.mma"', '"ame.other"')
AME_TWO_OPERANDS = """module {
  func.func @main(%a: memref<64x128xf16>, %b: memref<128x64xf16>) {
    "ame.mma"(%a, %b) : (memref<64x128xf16>, memref<128x64xf16>) -> ()
    return
  }
}
"""


def _confirmed_profile(**overrides):
    raw = {
        "schema_version": 1, "name": "ame-test", "version": "1", "status": "confirmed",
        "allowed_dialects": ["builtin", "func", "ame", "memref", "arith"],
        "forbidden_dialects": ["gpu"], "required_dialects": [],
        "entry_symbol": "main",
        "dialects": {"ame": {
            "status": "confirmed", "closed_world": True,
            "ops": {"ame.mma": {"status": "confirmed", "operands": 3,
                                "results": 0, "regions": 0}}}},
        "dtype_policy": {"status": "confirmed", "allowed": ["f16", "i64"]},
        "alignment_policy": None,
        "stage_rules": {},
        "pending": [], "notes": [],
    }
    raw.update(overrides)
    return raw


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _write_profile(tmp_path, raw, name="profile.json"):
    path = tmp_path / name
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


# ---------------------------------------------------------------- readers

def test_text_inventory_where():
    module = read_module(STAGE03, backend="text", verify=False)
    assert module.backend == "text"
    assert module.entry_symbols == ["main"]
    assert {"arith", "func", "linalg", "memref", "scf"} <= set(module.dialects())
    assert len(module.ops) > 100
    assert any("隐式 region" in limit for limit in module.limits)
    assert any(op.name == "func.return" for op in module.ops)


def test_official_inventory_where():
    pytest.importorskip("mlir.ir")
    module = read_module(STAGE03, backend="official", verify=True)
    names = {op.name for op in module.ops}
    assert "linalg.yield" in names          # implicit payload terminator
    assert "func.return" in names
    assert module.parse_mode == "strict"
    assert any(op.name == "func.func" and op.line for op in module.ops)
    assert module.diagnostics[0]["verified"] is True


def test_text_and_official_views_differ_only_by_documented_gaps():
    pytest.importorskip("mlir.ir")
    text = read_module(STAGE03, backend="text", verify=False)
    official = read_module(STAGE03, backend="official", verify=False)
    text_names = {op.name for op in text.ops}
    official_names = {op.name for op in official.ops}
    assert "linalg.yield" in official_names - text_names
    assert "func.return" in text_names


def test_unregistered_dialect_is_readable(tmp_path):
    path = _write(tmp_path, "ame.mlir", AME_MODULE)
    text = read_module(path, backend="text", verify=False)
    assert "ame" in text.dialects()
    pytest.importorskip("mlir.ir")
    official = read_module(path, backend="official", verify=False)
    assert official.parse_mode == "permissive"
    assert any(op.name == "ame.mma" for op in official.ops)


def test_auto_reports_text_fallback(tmp_path, monkeypatch):
    from tir_mlir_checker.dialect import reader
    path = _write(tmp_path, "ame.mlir", AME_MODULE)

    def unavailable(*args, **kwargs):
        raise DialectReadError("bindings unavailable in test", kind="dependency")

    monkeypatch.setattr(reader, "read_official", unavailable)
    module = reader.read_module(path, backend="auto")
    assert module.backend == "text"
    assert "bindings unavailable" in module.fallback_reason
    monkeypatch.setattr(reader, "read_official", unavailable)
    with pytest.raises(DialectReadError):
        reader.read_module(path, backend="official")


# ---------------------------------------------------------------- profiles

def test_generic_profile_accepts_where():
    result = check_module(STAGE03, profile="generic", backend="text", verify=False)
    assert result["tool_status"] == "ok"
    assert result["summary"]["mismatch"] == 0
    assert result["summary"]["unsupported"] == 0


def test_unknown_dialect_with_generic_profile_is_unsupported(tmp_path):
    path = _write(tmp_path, "ame.mlir", AME_MODULE)
    result = check_module(path, profile="generic", backend="text", verify=False)
    assert result["tool_status"] == "unsupported"
    assert result["exit_code"] == 4
    assert any(item["rule_id"] == "DIA-DIALECT-UNKNOWN" and item["status"] == "unsupported"
               for item in result["findings"])


def test_text_backend_does_not_claim_arity(tmp_path):
    path = _write(tmp_path, "two.mlir", AME_TWO_OPERANDS)
    profile = _write_profile(tmp_path, _confirmed_profile())
    result = check_module(path, profile_file=profile, backend="text", verify=False)
    assert not any(item["rule_id"] == "DIA-OP-ARITY" for item in result["findings"])
    assert result["tool_status"] == "ok"


def test_ame_candidate_marks_unknown_operations_uncertain(tmp_path):
    path = _write(tmp_path, "ame.mlir", AME_MODULE)
    result = check_module(path, profile="ame-candidate", backend="text", verify=False)
    unknown = [item for item in result["findings"] if item["rule_id"] == "DIA-OP-UNKNOWN"]
    assert unknown and all(item["status"] == "uncertain" for item in unknown)
    assert result["tool_status"] == "ok"
    assert result["semantic_conclusion"] == "bounded_checks_passed_with_pending"
    assert any(item["rule_id"] == "DIA-PROFILE-PENDING" for item in result["findings"])


def test_confirmed_profile_passes(tmp_path):
    path = _write(tmp_path, "ame.mlir", AME_MODULE)
    profile = _write_profile(tmp_path, _confirmed_profile())
    result = check_module(path, profile_file=profile, backend="text", verify=False)
    assert result["tool_status"] == "ok", result["findings"]
    assert not any(item["rule_id"] == "DIA-PROFILE-PENDING" for item in result["findings"])


def test_confirmed_profile_rejects_unknown_operation(tmp_path):
    path = _write(tmp_path, "ame.mlir", AME_UNKNOWN_OP)
    profile = _write_profile(tmp_path, _confirmed_profile())
    result = check_module(path, profile_file=profile, backend="text", verify=False)
    assert result["tool_status"] == "mismatch"
    assert result["exit_code"] == 1
    assert any(item["rule_id"] == "DIA-OP-UNKNOWN" and item["status"] == "mismatch"
               for item in result["findings"])


def test_dtype_policy_rejects_out_of_contract_dtype(tmp_path):
    path = _write(tmp_path, "ame.mlir", AME_F32_MODULE)
    profile = _write_profile(tmp_path, _confirmed_profile())
    result = check_module(path, profile_file=profile, backend="text", verify=False)
    assert any(item["rule_id"] == "DIA-TYPE-DTYPE" and item["status"] == "mismatch"
               for item in result["findings"])


def test_alignment_policy(tmp_path):
    module = """module {
  func.func @main() {
    %0 = memref.alloc() {alignment = 32 : i64} : memref<64xf16>
    return
  }
}
"""
    path = _write(tmp_path, "alloc.mlir", module)
    raw = _confirmed_profile(alignment_policy={
        "status": "confirmed", "alloc_ops": ["memref.alloc", "memref.alloca"],
        "required_alignment_bytes": 64})
    profile = _write_profile(tmp_path, raw)
    result = check_module(path, profile_file=profile, backend="text", verify=False)
    assert any(item["rule_id"] == "DIA-TYPE-ALIGN" and item["status"] == "mismatch"
               for item in result["findings"])


def test_entry_symbol_missing(tmp_path):
    module = AME_MODULE.replace("@main", "@kernel")
    path = _write(tmp_path, "ame.mlir", module)
    profile = _write_profile(tmp_path, _confirmed_profile())
    result = check_module(path, profile_file=profile, backend="text", verify=False)
    assert any(item["rule_id"] == "DIA-ENTRY-SYMBOL" and item["status"] == "mismatch"
               for item in result["findings"])


def test_forbidden_and_required_dialects(tmp_path):
    module = """module {
  func.func @main() {
    "gpu.launch"() : () -> ()
    return
  }
}
"""
    path = _write(tmp_path, "gpu.mlir", module)
    profile = _write_profile(tmp_path, _confirmed_profile(required_dialects=["ame"]))
    result = check_module(path, profile_file=profile, backend="text", verify=False)
    ids = {item["rule_id"] for item in result["findings"]}
    assert "DIA-DIALECT-FORBIDDEN" in ids
    assert "DIA-DIALECT-REQUIRED" in ids


def test_stage_policy(tmp_path):
    path = _write(tmp_path, "ame.mlir", AME_MODULE)
    profile = _write_profile(tmp_path, _confirmed_profile(stage_rules={"03": ["builtin", "func"]}))
    result = check_module(path, profile_file=profile, backend="text", stage="03", verify=False)
    assert any(item["rule_id"] == "DIA-STAGE-POLICY" and item["status"] == "mismatch"
               for item in result["findings"])


def test_symbol_resolution(tmp_path):
    module = """module {
  func.func @main() {
    func.call @missing() : () -> ()
    return
  }
}
"""
    path = _write(tmp_path, "call.mlir", module)
    result = check_module(path, profile="generic", backend="text", verify=False)
    assert any(item["rule_id"] == "DIA-SYMBOL-UNRESOLVED" and item["status"] == "mismatch"
               for item in result["findings"])


def test_required_attribute_is_uncertain_in_text_backend(tmp_path):
    module = AME_MODULE.replace('"ame.mma"(%a, %b, %c)',
                                '"ame.mma"(%a, %b, %c) {flags = 1 : i32}')
    path = _write(tmp_path, "attr.mlir", module)
    raw = _confirmed_profile()
    raw["dialects"]["ame"]["ops"]["ame.mma"]["attributes"] = ["mode"]
    profile = _write_profile(tmp_path, raw)
    result = check_module(path, profile_file=profile, backend="text", verify=False)
    attr = [item for item in result["findings"] if item["rule_id"] == "DIA-OP-ATTR"]
    assert attr and all(item["status"] == "uncertain" for item in attr)


def test_official_arity_check(tmp_path):
    pytest.importorskip("mlir.ir")
    path = _write(tmp_path, "two.mlir", AME_TWO_OPERANDS)
    profile = _write_profile(tmp_path, _confirmed_profile())
    result = check_module(path, profile_file=profile, backend="official", verify=False)
    assert any(item["rule_id"] == "DIA-OP-ARITY" and item["status"] == "mismatch"
               for item in result["findings"])


# ---------------------------------------------------------------- template / API

def test_template_roundtrip(tmp_path):
    path = _write(tmp_path, "ame.mlir", AME_MODULE)
    module = read_module(path, backend="text", verify=False)
    raw = template_from_module(module, namespace="ame")
    profile_from_dict(raw)  # must validate
    assert raw["status"] == "pending"
    assert raw["dialects"]["ame"]["closed_world"] is True
    assert raw["dialects"]["ame"]["ops"]["ame.mma"]["status"] == "pending"
    saved = _write_profile(tmp_path, raw, name="template.json")
    loaded = load_profile(saved)
    assert loaded.name == raw["name"]


def test_check_trace_stage_metadata():
    result = check_trace(WHERE, stage="03", profile="generic", backend="text", verify=False)
    assert result["stage"] == "03_mlir_linalg"
    assert result["trace"]["run_id"]
    assert result["tool_status"] == "ok"


def test_artifact_output(tmp_path):
    path = _write(tmp_path, "ame.mlir", AME_MODULE)
    out = tmp_path / "run-001"
    result = check_module(path, profile="ame-candidate", backend="text",
                          verify=False, out=out)
    assert (out / "dialect-result.json").is_file()
    report = (out / "dialect-report.md").read_text(encoding="utf-8")
    assert "## 1. 检查结论" in report and "## 3. 问题定位与判断" in report
    assert result["artifacts"]["result"].endswith("dialect-result.json")


def test_finding_schema_matches_host_diagnostics(tmp_path):
    path = _write(tmp_path, "ame.mlir", AME_MODULE)
    result = check_module(path, profile="ame-candidate", backend="text", verify=False)
    required = {"rule_id", "status", "method", "message", "locations", "details", "scope"}
    for item in result["findings"]:
        assert required <= set(item)
        for location in item["locations"]:
            assert "file" in location


# ---------------------------------------------------------------- CLI

def test_cli_check(tmp_path, capsys):
    from tir_mlir_checker.dialect.cli import main
    path = _write(tmp_path, "ame.mlir", AME_MODULE)
    code = main(["check", "--module", str(path), "--profile", "ame-candidate",
                 "--backend", "text", "--no-verify"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0 and payload["exit_code"] == 0
    assert payload["layer"] == "dialect"
    assert payload["profile"]["name"] == "ame-candidate"


def test_cli_missing_module_is_input_error(capsys):
    from tir_mlir_checker.dialect.cli import main
    code = main(["check", "--module", "/nonexistent/x.mlir", "--backend", "text"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 2 and payload["tool_status"] == "input_error"

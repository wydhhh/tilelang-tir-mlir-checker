"""MLIR Dialect-layer contract checking (project extension).

This subpackage is independent from the host checker's IR-level modules.  It
reads MLIR modules as dialect-qualified operation facts, then evaluates a
target contract profile (builtin ``generic``, ``ame-candidate``, or an
external JSON profile).  It does not execute IR or prove semantic equivalence.

Public API::

    from tir_mlir_checker.dialect import check_module, check_trace, inventory_module
"""
from __future__ import annotations

from pathlib import Path

from ..trace_io import STAGES, InputError, load_trace
from .analyzers import analyze_module, observed_dtypes
from .model import (STATUSES, AlignmentPolicy, ContractProfile, DialectContract,
                    DialectModule, DialectOp, DialectReadError, DtypePolicy,
                    OpContract, ProfileError, op_contract_status)
from .profile import load_profile, profile_from_dict, profile_to_dict, template_from_module
from .reader import describe, read_module
from .report import build_result, write_outputs

__all__ = [
    "check_module", "check_trace", "inventory_module", "load_profile",
    "read_module", "analyze_module", "template_from_module", "profile_to_dict",
    "DialectModule", "DialectOp", "ContractProfile", "DialectContract",
    "OpContract", "DtypePolicy", "AlignmentPolicy", "DialectReadError",
    "ProfileError", "STATUSES", "op_contract_status",
]


def _stage_index(stage):
    value = str(stage)
    if value in STAGES:
        return STAGES.index(value), value
    try:
        index = int(value)
    except ValueError as exc:
        raise InputError(f"未知阶段：{stage}") from exc
    if not 0 <= index < len(STAGES):
        raise InputError(f"未知阶段：{stage}")
    return index, STAGES[index]


def _resolve_profile(profile, profile_file):
    return load_profile(profile_file or profile)


def check_module(module=None, *, text=None, profile="generic", profile_file=None,
                 backend="auto", stage=None, verify=True, out=None):
    """Check one module (path or text) against a contract profile."""
    contract = _resolve_profile(profile, profile_file)
    facts = read_module(module, text=text, backend=backend, verify=verify)
    findings = analyze_module(facts, contract, stage=stage)
    result = build_result(facts, contract, findings, stage=stage)
    if out is not None:
        directory = Path(out).expanduser()
        result["artifacts"] = {
            "result": str(directory / "dialect-result.json"),
            "report": str(directory / "dialect-report.md"),
        }
        write_outputs(result, directory)
    return result


def check_trace(trace_dir, *, stage="03", profile="generic", profile_file=None,
                backend="auto", verify=True, out=None):
    """Check one stage of an existing five-stage Trace against a profile."""
    index, key = _stage_index(stage)
    group = load_trace(trace_dir)
    record = group.stages[index]
    contract = _resolve_profile(profile, profile_file)
    facts = read_module(record["file"], text=group.text[key], backend=backend, verify=verify)
    findings = analyze_module(facts, contract, stage=key)
    result = build_result(facts, contract, findings, stage=key)
    result["trace"] = {"root": str(group.root), "run_id": group.run_id,
                       "stage": key, "origin": record.get("origin")}
    if out is not None:
        directory = Path(out).expanduser()
        result["artifacts"] = {
            "result": str(directory / "dialect-result.json"),
            "report": str(directory / "dialect-report.md"),
        }
        write_outputs(result, directory)
    return result


def inventory_module(module=None, *, text=None, backend="auto", verify=True, out=None):
    """List dialects, operations, symbols and limits observed in a module."""
    facts = read_module(module, text=text, backend=backend, verify=verify)
    result = describe(facts)
    result["dtypes"] = sorted(observed_dtypes(facts))
    if out is not None:
        import json
        directory = Path(out).expanduser()
        if directory.exists():
            raise FileExistsError(f"输出目录必须不存在：{directory}")
        directory.mkdir(parents=True)
        result["artifacts"] = {"inventory": str(directory / "dialect-inventory.json")}
        (directory / "dialect-inventory.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result

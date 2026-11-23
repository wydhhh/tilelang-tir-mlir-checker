"""Load, validate, serialize and synthesize contract profiles."""
from __future__ import annotations

import json
from pathlib import Path

from .model import (STATUSES, AlignmentPolicy, ContractProfile, DialectContract,
                    DtypePolicy, OpContract, ProfileError)
from .profiles import BUILTIN_PROFILES, STANDARD_DIALECTS


def _require(condition, message):
    if not condition:
        raise ProfileError(message)


def _string_list(value, field, *, allow_none=False):
    if value is None and allow_none:
        return None
    _require(isinstance(value, list) and all(isinstance(v, str) for v in value),
             f"{field} 必须是字符串数组")
    return tuple(value)


def _status(value, field):
    _require(value in STATUSES, f"{field} 必须是 {STATUSES} 之一，实际为 {value!r}")
    return value


def _optional_count(value, field):
    _require(value is None or (type(value) is int and value >= 0),
             f"{field} 必须是非负整数或 null")
    return value


def _op_contract(name, raw):
    _require(isinstance(raw, dict), f"dialects.*.ops.{name} 必须是对象")
    unknown = set(raw) - {"status", "operands", "results", "regions", "attributes",
                          "allowed_attributes", "note"}
    _require(not unknown, f"dialects.*.ops.{name} 含未知字段：{sorted(unknown)}")
    return OpContract(
        name=name,
        status=_status(raw.get("status", "candidate"), f"ops.{name}.status"),
        operands=_optional_count(raw.get("operands"), f"ops.{name}.operands"),
        results=_optional_count(raw.get("results"), f"ops.{name}.results"),
        regions=_optional_count(raw.get("regions"), f"ops.{name}.regions"),
        attributes=_string_list(raw.get("attributes", []), f"ops.{name}.attributes"),
        allowed_attributes=_string_list(raw.get("allowed_attributes"), f"ops.{name}.allowed_attributes",
                                        allow_none=True),
        note=str(raw.get("note", "")),
    )


def _dialect_contract(namespace, raw):
    _require(isinstance(raw, dict), f"dialects.{namespace} 必须是对象")
    unknown = set(raw) - {"status", "closed_world", "ops", "allowed_dtypes",
                          "attributes_required", "attributes_allowed", "note"}
    _require(not unknown, f"dialects.{namespace} 含未知字段：{sorted(unknown)}")
    ops_raw = raw.get("ops", {})
    _require(isinstance(ops_raw, dict), f"dialects.{namespace}.ops 必须是对象")
    ops = {name: _op_contract(name, entry) for name, entry in ops_raw.items()}
    for name in ops:
        _require(name.startswith(namespace + "."),
                 f"dialects.{namespace}.ops 中的 {name} 不属于该命名空间")
    return DialectContract(
        namespace=namespace,
        status=_status(raw.get("status", "candidate"), f"dialects.{namespace}.status"),
        closed_world=bool(raw.get("closed_world", False)),
        ops=ops,
        allowed_dtypes=_string_list(raw.get("allowed_dtypes"), f"dialects.{namespace}.allowed_dtypes",
                                    allow_none=True),
        attributes_required=_string_list(raw.get("attributes_required", []),
                                         f"dialects.{namespace}.attributes_required"),
        attributes_allowed=_string_list(raw.get("attributes_allowed"),
                                        f"dialects.{namespace}.attributes_allowed", allow_none=True),
        note=str(raw.get("note", "")),
    )


def profile_from_dict(raw):
    """Validate one JSON-shaped profile and return a ContractProfile."""
    _require(isinstance(raw, dict), "profile 必须是 JSON 对象")
    allowed_keys = {"schema_version", "name", "version", "status", "description",
                    "allowed_dialects", "forbidden_dialects", "required_dialects",
                    "entry_symbol", "dialects", "dtype_policy", "alignment_policy",
                    "stage_rules", "pending", "notes"}
    unknown = set(raw) - allowed_keys
    _require(not unknown, f"profile 含未知字段：{sorted(unknown)}")
    _require(raw.get("schema_version") == 1, "仅支持 schema_version=1")
    _require(isinstance(raw.get("name"), str) and raw["name"], "name 必须是非空字符串")
    _require(isinstance(raw.get("version"), str) and raw["version"], "version 必须是非空字符串")
    status = _status(raw.get("status", "candidate"), "status")

    dialects_raw = raw.get("dialects", {})
    _require(isinstance(dialects_raw, dict), "dialects 必须是对象")
    dialects = {ns: _dialect_contract(ns, entry) for ns, entry in dialects_raw.items()}

    dtype_policy = None
    if raw.get("dtype_policy") is not None:
        item = raw["dtype_policy"]
        _require(isinstance(item, dict), "dtype_policy 必须是对象或 null")
        _require(set(item) <= {"status", "allowed", "note"},
                 f"dtype_policy 含未知字段：{sorted(set(item) - {'status', 'allowed', 'note'})}")
        dtype_policy = DtypePolicy(
            status=_status(item.get("status", "pending"), "dtype_policy.status"),
            allowed=tuple(_string_list(item.get("allowed", []), "dtype_policy.allowed")),
            note=str(item.get("note", "")))

    alignment_policy = None
    if raw.get("alignment_policy") is not None:
        item = raw["alignment_policy"]
        _require(isinstance(item, dict), "alignment_policy 必须是对象或 null")
        _require(set(item) <= {"status", "alloc_ops", "required_alignment_bytes", "note"},
                 f"alignment_policy 含未知字段：{sorted(set(item) - {'status', 'alloc_ops', 'required_alignment_bytes', 'note'})}")
        required = item.get("required_alignment_bytes")
        _require(required is None or (type(required) is int and required > 0),
                 "required_alignment_bytes 必须是正整数或 null")
        alignment_policy = AlignmentPolicy(
            status=_status(item.get("status", "pending"), "alignment_policy.status"),
            alloc_ops=tuple(_string_list(item.get("alloc_ops", ["memref.alloc", "memref.alloca"]),
                                         "alignment_policy.alloc_ops")),
            required_alignment_bytes=required,
            note=str(item.get("note", "")))

    stage_rules_raw = raw.get("stage_rules", {})
    _require(isinstance(stage_rules_raw, dict), "stage_rules 必须是对象")
    stage_rules = {}
    for stage, names in stage_rules_raw.items():
        _require(isinstance(stage, str) and stage, "stage_rules 的键必须是非空字符串")
        stage_rules[stage] = frozenset(_string_list(names, f"stage_rules.{stage}"))

    entry = raw.get("entry_symbol")
    _require(entry is None or (isinstance(entry, str) and entry), "entry_symbol 必须是非空字符串或 null")

    return ContractProfile(
        name=raw["name"], version=raw["version"], status=status,
        allowed_dialects=frozenset(_string_list(raw.get("allowed_dialects", []),
                                                "allowed_dialects")),
        forbidden_dialects=frozenset(_string_list(raw.get("forbidden_dialects", []),
                                                  "forbidden_dialects")),
        required_dialects=frozenset(_string_list(raw.get("required_dialects", []),
                                                 "required_dialects")),
        entry_symbol=entry, dialects=dialects,
        dtype_policy=dtype_policy, alignment_policy=alignment_policy,
        stage_rules=stage_rules,
        pending=tuple(_string_list(raw.get("pending", []), "pending")),
        notes=tuple(_string_list(raw.get("notes", []), "notes")),
    )


def load_profile(name_or_path):
    """Resolve a builtin name or a JSON file path to a ContractProfile."""
    text = str(name_or_path)
    raw = BUILTIN_PROFILES.get(text)
    if raw is None:
        path = Path(text).expanduser()
        _require(path.is_file(), f"profile 既不是内置名，也不是可读文件：{text}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise ProfileError(f"无法读取 profile {path}：{exc}") from exc
    return profile_from_dict(raw)


def profile_to_dict(profile):
    """Serialize a ContractProfile back to the JSON schema."""
    raw = {
        "schema_version": 1, "name": profile.name, "version": profile.version,
        "status": profile.status,
        "allowed_dialects": sorted(profile.allowed_dialects),
        "forbidden_dialects": sorted(profile.forbidden_dialects),
        "required_dialects": sorted(profile.required_dialects),
        "entry_symbol": profile.entry_symbol,
        "dialects": {
            ns: {
                "status": item.status, "closed_world": item.closed_world,
                "ops": {name: {key: value for key, value in {
                    "status": op.status, "operands": op.operands,
                    "results": op.results, "regions": op.regions,
                    "attributes": list(op.attributes),
                    "allowed_attributes": None if op.allowed_attributes is None
                    else list(op.allowed_attributes),
                    "note": op.note}.items() if value not in (None, [], "")}
                    for name, op in item.ops.items()},
                "allowed_dtypes": None if item.allowed_dtypes is None else list(item.allowed_dtypes),
                "attributes_required": list(item.attributes_required),
                "attributes_allowed": None if item.attributes_allowed is None
                else list(item.attributes_allowed),
                "note": item.note,
            } for ns, item in profile.dialects.items()
        },
        "dtype_policy": None if profile.dtype_policy is None else {
            "status": profile.dtype_policy.status,
            "allowed": list(profile.dtype_policy.allowed),
            "note": profile.dtype_policy.note},
        "alignment_policy": None if profile.alignment_policy is None else {
            "status": profile.alignment_policy.status,
            "alloc_ops": list(profile.alignment_policy.alloc_ops),
            "required_alignment_bytes": profile.alignment_policy.required_alignment_bytes,
            "note": profile.alignment_policy.note},
        "stage_rules": {stage: sorted(names) for stage, names in profile.stage_rules.items()},
        "pending": list(profile.pending), "notes": list(profile.notes),
    }
    return raw


def template_from_module(module, *, namespace, name=None, version="0.1", status="pending"):
    """Turn an observed module into an editable contract skeleton.

    Everything observed under ``namespace`` is emitted as ``pending``: the
    template records inventory, it does not claim a contract.
    """
    ops = {}
    for op in module.ops_by_dialect(namespace):
        ops.setdefault(op.name, {
            "status": "pending",
            "note": f"observed at {op.node_path or ('line ' + str(op.line))}; "
                    "operand/result/region schema pending vendor documentation",
        })
    observed = sorted({op.dialect for op in module.ops} - {namespace})
    dtypes = sorted(_observed_dtypes(module))
    return {
        "schema_version": 1,
        "name": name or f"{namespace}-template",
        "version": version,
        "status": status,
        "description": f"Skeleton generated from {module.path}; every item is pending.",
        "allowed_dialects": [ns for ns in observed if ns in STANDARD_DIALECTS],
        "forbidden_dialects": [],
        "required_dialects": [namespace] if ops else [],
        "entry_symbol": module.entry_symbols[0] if len(module.entry_symbols) == 1 else None,
        "dialects": {
            namespace: {
                "status": status,
                "closed_world": True,
                "ops": ops,
                "allowed_dtypes": dtypes or None,
                "attributes_required": [],
                "attributes_allowed": None,
                "note": "Generated from an observed module; confirm against vendor docs.",
            }
        },
        "dtype_policy": {"status": status, "allowed": dtypes, "note": "Observed dtypes; unconfirmed."}
        if dtypes else None,
        "alignment_policy": None,
        "stage_rules": {},
        "pending": [f"{namespace} 方言契约尚未确认；本模板由实际模块生成"],
        "notes": ["template 只记录现状，不作为通过依据。"],
    }


def _observed_dtypes(module):
    from .analyzers import observed_dtypes
    return observed_dtypes(module)

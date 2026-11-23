"""Contract analyzers: pure functions from (module, profile, stage) to findings.

Every finding uses the host checker's ``diagnostics.finding`` schema, so the
dialect layer can be reported next to the existing IR-level findings.
"""
from __future__ import annotations

import re

from ..diagnostics import finding
from .profiles import STANDARD_DIALECTS
from .text_reader import SCALAR_TYPES

SCOPE = "当前模块的方言/操作结构与 profile 条款；不执行数值，不证明语义等价。"


def _status(policy_status):
    return "mismatch" if policy_status == "confirmed" else "uncertain"


def _first(module, dialect=None, name=None):
    for op in module.ops:
        if (dialect is None or op.dialect == dialect) and (name is None or op.name == name):
            return op
    return module.ops[0] if module.ops else None


def _locations(module, op, stage):
    return [op.location(stage)] if op is not None else []


def dtypes_in_type(value):
    found = set()
    if value in SCALAR_TYPES:
        found.add(value)
    for match in re.finditer(r"(?:memref|tensor|vector)<([^<>]*)>", value):
        inner = match.group(1)
        last = inner.split("x")[-1].strip()
        if last in SCALAR_TYPES:
            found.add(last)
    for match in re.finditer(r"x([A-Za-z]\w*)>", value):
        if match.group(1) in SCALAR_TYPES:
            found.add(match.group(1))
    for match in re.finditer(r"complex<([^<>]*)>", value):
        if match.group(1) in SCALAR_TYPES:
            found.add(match.group(1))
    return found


def observed_dtypes(module, ops=None):
    found = set()
    for op in module.ops if ops is None else ops:
        for type_string in op.types:
            found.update(dtypes_in_type(type_string))
    return frozenset(found)


def check_dialects(module, profile, stage):
    findings = []
    seen = module.dialects()
    known = STANDARD_DIALECTS | set(profile.dialects) | profile.allowed_dialects
    known = known | profile.forbidden_dialects
    for dialect in seen:
        if dialect in known:
            continue
        op = _first(module, dialect=dialect)
        findings.append(finding(
            "DIA-DIALECT-UNKNOWN", "unsupported", "dialect-contract",
            f"未登记方言 {dialect}：既不在内置标准方言集中，也无 profile 条款",
            stage=stage, locations=_locations(module, op, stage),
            details={"dialect": dialect, "backend": module.backend},
            scope="未知方言只报告不判定；未知不等于错误。",
            next_action="在 profile 中登记该方言（命名空间/操作/属性）后重跑。"))
    for dialect in sorted(set(seen) & profile.forbidden_dialects):
        op = _first(module, dialect=dialect)
        findings.append(finding(
            "DIA-DIALECT-FORBIDDEN", _status(profile.status), "dialect-contract",
            f"出现 profile 禁止的方言 {dialect}",
            stage=stage, locations=_locations(module, op, stage),
            details={"dialect": dialect, "profile_status": profile.status},
            scope=SCOPE))
    for dialect in sorted(profile.required_dialects - set(seen)):
        findings.append(finding(
            "DIA-DIALECT-REQUIRED", _status(profile.status), "dialect-contract",
            f"profile 要求但缺失的方言 {dialect}",
            stage=stage, locations=_locations(module, _first(module), stage),
            details={"dialect": dialect, "profile_status": profile.status},
            scope=SCOPE))
    if profile.allowed_dialects:
        for dialect in sorted((set(seen) & known) - profile.allowed_dialects
                              - profile.forbidden_dialects):
            op = _first(module, dialect=dialect)
            findings.append(finding(
                "DIA-DIALECT-NOT-ALLOWED", _status(profile.status), "dialect-contract",
                f"方言 {dialect} 不在 profile 的允许集合内",
                stage=stage, locations=_locations(module, op, stage),
                details={"dialect": dialect, "profile_status": profile.status},
                scope=SCOPE))
    return findings


def check_ops(module, profile, stage):
    findings = []
    for namespace, contract in profile.dialects.items():
        ops = module.ops_by_dialect(namespace)
        if not ops:
            continue
        if contract.closed_world:
            for op in ops:
                if op.name in contract.ops:
                    continue
                findings.append(finding(
                    "DIA-OP-UNKNOWN", _status(contract.status), "dialect-contract",
                    f"闭合世界方言 {namespace} 中出现契约外的操作 {op.name}",
                    stage=stage, locations=_locations(module, op, stage),
                    details={"dialect": namespace, "operation": op.name,
                             "profile_status": contract.status, "backend": module.backend},
                    scope=SCOPE,
                    next_action="确认该操作是否属于契约；未确认前只能作为待复核。"))
        for op in ops:
            spec = contract.ops.get(op.name)
            if spec is None:
                continue
            status = _status(spec.status)
            if module.backend == "official":
                for label, expected, actual in (
                        ("operands", spec.operands, len(op.operands)),
                        ("results", spec.results, len(op.results)),
                        ("regions", spec.regions, op.region_count)):
                    if expected is not None and actual != expected:
                        findings.append(finding(
                            "DIA-OP-ARITY", status, "dialect-contract",
                            f"{op.name} 的 {label} 数量为 {actual}，契约为 {expected}",
                            stage=stage, locations=_locations(module, op, stage),
                            details={"operation": op.name, "field": label,
                                     "expected": expected, "actual": actual,
                                     "profile_status": spec.status},
                            scope=SCOPE))
            missing = [name for name in spec.attributes if name not in op.attributes]
            if missing:
                findings.append(finding(
                    "DIA-OP-ATTR", status if module.backend == "official" else "uncertain",
                    "dialect-contract",
                    f"{op.name} 缺少契约要求的属性：{', '.join(missing)}",
                    stage=stage, locations=_locations(module, op, stage),
                    details={"operation": op.name, "missing": missing,
                             "profile_status": spec.status, "backend": module.backend},
                    scope=SCOPE if module.backend == "official"
                    else SCOPE + "（text 后端的属性解析为近似值）"))
            if spec.allowed_attributes is not None:
                extra = sorted(set(op.attributes) - set(spec.allowed_attributes))
                if extra:
                    findings.append(finding(
                        "DIA-OP-ATTR", status if module.backend == "official" else "uncertain",
                        "dialect-contract",
                        f"{op.name} 出现契约属性集之外的属性：{', '.join(extra)}",
                        stage=stage, locations=_locations(module, op, stage),
                        details={"operation": op.name, "unexpected": extra,
                                 "profile_status": spec.status, "backend": module.backend},
                        scope=SCOPE))
        if contract.attributes_required or contract.attributes_allowed is not None:
            for op in ops:
                missing = [name for name in contract.attributes_required
                           if name not in op.attributes]
                if missing:
                    findings.append(finding(
                        "DIA-OP-ATTR", _status(contract.status)
                        if module.backend == "official" else "uncertain", "dialect-contract",
                        f"{op.name} 缺少方言级必填属性：{', '.join(missing)}",
                        stage=stage, locations=_locations(module, op, stage),
                        details={"operation": op.name, "missing": missing,
                                 "dialect": namespace, "backend": module.backend},
                        scope=SCOPE))
                if contract.attributes_allowed is not None:
                    extra = sorted(set(op.attributes) - set(contract.attributes_allowed))
                    if extra:
                        findings.append(finding(
                            "DIA-OP-ATTR", _status(contract.status)
                            if module.backend == "official" else "uncertain",
                            "dialect-contract",
                            f"{op.name} 出现方言级属性集之外的属性：{', '.join(extra)}",
                            stage=stage, locations=_locations(module, op, stage),
                            details={"operation": op.name, "unexpected": extra,
                                     "dialect": namespace, "backend": module.backend},
                            scope=SCOPE))
    return findings


def check_types(module, profile, stage):
    findings = []
    if profile.dtype_policy is not None and profile.dtype_policy.allowed:
        allowed = set(profile.dtype_policy.allowed)
        observed = observed_dtypes(module) - {"index"}
        for dtype in sorted(observed - allowed):
            example = next((op for op in module.ops
                            if dtype in observed_dtypes(module, [op])), _first(module))
            findings.append(finding(
                "DIA-TYPE-DTYPE", _status(profile.dtype_policy.status), "dialect-contract",
                f"数据类型 {dtype} 不在 dtype 策略允许集合内",
                stage=stage, locations=_locations(module, example, stage),
                details={"dtype": dtype, "allowed": sorted(allowed),
                         "profile_status": profile.dtype_policy.status},
                scope=SCOPE))
    for namespace, contract in profile.dialects.items():
        if contract.allowed_dtypes is None:
            continue
        allowed = set(contract.allowed_dtypes)
        ops = module.ops_by_dialect(namespace)
        observed = observed_dtypes(module, ops) - {"index"}
        for dtype in sorted(observed - allowed):
            example = next((op for op in ops
                            if dtype in observed_dtypes(module, [op])), None)
            findings.append(finding(
                "DIA-TYPE-DTYPE", _status(contract.status), "dialect-contract",
                f"方言 {namespace} 出现契约外的数据类型 {dtype}",
                stage=stage, locations=_locations(module, example, stage),
                details={"dialect": namespace, "dtype": dtype,
                         "allowed": sorted(allowed), "profile_status": contract.status},
                scope=SCOPE))
    return findings


def check_alignment(module, profile, stage):
    findings = []
    policy = profile.alignment_policy
    if policy is None or policy.required_alignment_bytes is None:
        return findings
    required = policy.required_alignment_bytes
    for op in module.ops:
        if op.name not in policy.alloc_ops:
            continue
        raw = op.attributes.get("alignment")
        match = re.match(r"\s*(\d+)", raw or "")
        actual = int(match.group(1)) if match else None
        if actual is None or actual < required or actual & (actual - 1):
            findings.append(finding(
                "DIA-TYPE-ALIGN", _status(policy.status), "dialect-contract",
                f"{op.name} 的对齐为 {actual}，契约要求 >= {required} 且为 2 的幂",
                stage=stage, locations=_locations(module, op, stage),
                details={"operation": op.name, "actual": actual, "required": required,
                         "profile_status": policy.status, "backend": module.backend},
                scope=SCOPE))
    return findings


def check_entry(module, profile, stage):
    findings = []
    if not profile.entry_symbol:
        return findings
    defs = [name for name in module.symbol_defs if name == profile.entry_symbol]
    if not defs:
        findings.append(finding(
            "DIA-ENTRY-SYMBOL", _status(profile.status), "dialect-contract",
            f"契约要求的入口符号 @{profile.entry_symbol} 不存在",
            stage=stage, locations=_locations(module, _first(module), stage),
            details={"entry": profile.entry_symbol, "defined": module.entry_symbols,
                     "profile_status": profile.status},
            scope=SCOPE))
    elif len(defs) > 1:
        findings.append(finding(
            "DIA-ENTRY-SYMBOL", "mismatch", "dialect-contract",
            f"入口符号 @{profile.entry_symbol} 定义了 {len(defs)} 次",
            stage=stage, locations=_locations(module, _first(module, name="func.func"), stage),
            details={"entry": profile.entry_symbol, "count": len(defs)},
            scope=SCOPE))
    return findings


def check_symbols(module, profile, stage):
    findings = []
    defined = set(module.symbol_defs)
    for callee in sorted(set(module.callees) - defined):
        op = _first(module, name="func.call")
        findings.append(finding(
            "DIA-SYMBOL-UNRESOLVED", "mismatch", "symbol-resolution",
            f"func.call 的目标 @{callee} 在模块内未定义",
            stage=stage, locations=_locations(module, op, stage),
            details={"callee": callee, "defined": sorted(defined)},
            scope="模块内符号解析是结构事实；外部声明应在模块内显式出现。"))
    return findings


def check_stage(module, profile, stage):
    findings = []
    if not stage or not profile.stage_rules:
        return findings
    allowed = profile.stage_rules.get(stage) or profile.stage_rules.get(str(stage)[:2])
    if allowed is None:
        return findings
    for dialect in sorted(set(module.dialects()) - allowed):
        op = _first(module, dialect=dialect)
        findings.append(finding(
            "DIA-STAGE-POLICY", _status(profile.status), "dialect-contract",
            f"阶段 {stage} 不允许方言 {dialect}",
            stage=stage, locations=_locations(module, op, stage),
            details={"stage": stage, "dialect": dialect, "allowed": sorted(allowed),
                     "profile_status": profile.status},
            scope=SCOPE))
    return findings


def check_verifier(module, stage):
    findings = []
    for record in module.diagnostics:
        if record.get("check") != "verifier":
            continue
        verified = record.get("verified")
        if verified is True:
            continue
        messages = [item.get("message", str(item)) for item in module.diagnostics
                    if isinstance(item, dict) and "check" not in item and "message" in item]
        status = "mismatch" if verified is False else "uncertain"
        findings.append(finding(
            "DIA-VERIFIER", status, "official-verifier",
            "官方 MLIR verifier " + ("拒绝该模块" if verified is False else "未能完成"),
            stage=stage, locations=_locations(module, _first(module), stage),
            details={"verified": verified, "messages": messages[:20],
                     "error": record.get("error")},
            scope="官方 verifier 只回答结构良构；不证明跨阶段语义等价。"))
    return findings


def pending_items(profile):
    items = []
    if profile.status != "confirmed":
        items.append(f"profile {profile.name}:{profile.version} 整体状态={profile.status}")
    for namespace, contract in sorted(profile.dialects.items()):
        if contract.status != "confirmed":
            items.append(f"方言 {namespace}: {contract.status}"
                         + (f"（{contract.note}）" if contract.note else ""))
        for name, spec in sorted(contract.ops.items()):
            if spec.status != "confirmed":
                items.append(f"操作 {name}: {spec.status}")
    if profile.dtype_policy is not None and profile.dtype_policy.status != "confirmed":
        items.append(f"dtype 策略: {profile.dtype_policy.status}")
    if profile.alignment_policy is not None and profile.alignment_policy.status != "confirmed":
        items.append(f"对齐策略: {profile.alignment_policy.status}")
    items.extend(profile.pending)
    return items


def check_pending(profile, stage):
    items = pending_items(profile)
    if not items:
        return []
    return [finding(
        "DIA-PROFILE-PENDING", "uncertain", "dialect-contract",
        f"契约仍有 {len(items)} 项未确认条款",
        stage=stage, details={"count": len(items), "pending": items[:100]},
        scope="未确认条款既不构成通过，也不构成错误。",
        next_action="以厂商文档确认后更新 profile；不要为取得通过而删除 pending 项。")]


def analyze_module(module, profile, *, stage=None):
    findings = []
    findings += check_dialects(module, profile, stage)
    findings += check_ops(module, profile, stage)
    findings += check_types(module, profile, stage)
    findings += check_alignment(module, profile, stage)
    findings += check_entry(module, profile, stage)
    findings += check_symbols(module, profile, stage)
    findings += check_stage(module, profile, stage)
    findings += check_verifier(module, stage)
    findings += check_pending(profile, stage)
    return findings

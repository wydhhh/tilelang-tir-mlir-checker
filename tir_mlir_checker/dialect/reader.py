"""Backend selection: official MLIR objects when available, text otherwise.

``auto`` prefers the official backend and falls back to text *with the reason
recorded* in the module; a forced backend never silently switches.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .model import DialectModule, DialectReadError
from .official_reader import read_official
from .text_reader import read_text


def _source(path, text):
    if text is not None:
        return text, None
    if path is None:
        raise DialectReadError("必须提供 module 路径或模块文本", kind="input")
    source = Path(path).expanduser()
    try:
        raw = source.read_bytes()
        return raw.decode("utf-8"), raw
    except (OSError, UnicodeError) as exc:
        raise DialectReadError(f"模块不可读：{source}：{exc}", kind="input") from exc


def read_module(path=None, *, text=None, backend="auto", verify=True):
    """Read one MLIR module into a DialectModule."""
    if backend not in {"auto", "official", "text"}:
        raise DialectReadError(f"未知后端：{backend}", kind="input")
    value, raw = _source(path, text)
    digest = hashlib.sha256(raw if raw is not None else value.encode()).hexdigest()
    name = str(Path(path).expanduser().resolve()) if path is not None else "<text>"

    if backend == "text":
        return read_text(value, path=name, sha256=digest)
    if backend == "official":
        return read_official(value, path=name, sha256=digest, verify=verify)

    try:
        return read_official(value, path=name, sha256=digest, verify=verify)
    except DialectReadError as exc:
        module = read_text(value, path=name, sha256=digest)
        module.fallback_reason = f"official 后端不可用（{exc.kind}）：{exc}"
        module.limits.append("已回退到 text 后端；结论强度按 text 边界降低，回退原因见 fallback_reason")
        return module


def describe(module):
    """Compact facts for the inventory command."""
    dialects = {}
    for op in module.ops:
        entry = dialects.setdefault(op.dialect, {"ops": {}, "op_count": 0})
        entry["op_count"] += 1
        entry["ops"][op.name] = entry["ops"].get(op.name, 0) + 1
    return {
        "tool_status": "ok", "exit_code": 0, "layer": "dialect", "mode": "inventory",
        "module": module.public(),
        "dialects": {name: {"op_count": item["op_count"],
                            "ops": dict(sorted(item["ops"].items()))}
                     for name, item in sorted(dialects.items())},
        "symbol_defs": sorted(set(module.symbol_defs)),
        "callees": sorted(set(module.callees)),
        "module_attributes": module.module_attributes,
        "diagnostics": module.diagnostics,
        "limits": module.limits,
    }

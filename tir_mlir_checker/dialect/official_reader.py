"""Official MLIR object reader.

When the Python bindings are available this is the strong backend: canonical
operation names (including implicit regions and terminators), exact operand,
result and region counts, official attributes, source locations, and an
optional verifier run.  It never parses printed assembly itself.
"""
from __future__ import annotations

import hashlib
import re

from .model import DialectModule, DialectOp, DialectReadError
from .profiles import STANDARD_DIALECTS
from .verify import verify_module

_UNREGISTERED_HINTS = ("unregistered", "unknown dialect", "not registered",
                       "custom op", "dialect")


def _split_symbol(value):
    value = str(value).strip()
    return value.strip('"').lstrip("@")


def _location(op):
    try:
        filename = op.location.filename
        line = int(op.location.start_line)
    except (ValueError, AttributeError):
        return None, str(op.location)
    if line > 0 and filename == "-":
        return line, str(op.location)
    return None, str(op.location)


def read_official(text, *, path="<text>", sha256=None, verify=True):
    try:
        import mlir.ir as ir
    except (ImportError, OSError) as exc:
        raise DialectReadError(f"MLIR Python bindings 不可用：{exc}", kind="dependency") from exc

    module = DialectModule(path=path, sha256=sha256 or hashlib.sha256(text.encode()).hexdigest(),
                           backend="official", parse_mode="strict")
    module.limits.append("official 后端读官方操作对象；隐式 region/terminator 可见，位置来自 op.location")
    records = []

    def handler(diagnostic):
        try:
            records.append({"message": str(diagnostic), "severity": str(diagnostic.severity),
                            "location": str(diagnostic.location)})
        except Exception:  # pragma: no cover - unusual binding builds
            records.append({"message": str(diagnostic)})
        return True

    context = ir.Context()
    with context:
        context.allow_unregistered_dialects = False
        try:
            context.load_all_available_dialects()
        except Exception:  # pragma: no cover - depends on build
            pass
        handle = context.attach_diagnostic_handler(handler)
        try:
            try:
                parsed = ir.Module.parse(text)
            except Exception as first:
                message = str(first)
                if any(hint in message.lower() for hint in _UNREGISTERED_HINTS):
                    module.parse_mode = "permissive"
                    context.allow_unregistered_dialects = True
                    records.append({"message": "strict 解析失败，按允许未注册方言重试：" + message,
                                    "severity": "remark"})
                    try:
                        parsed = ir.Module.parse(text)
                    except Exception as second:
                        raise DialectReadError(f"官方解析失败（含未注册方言重试）：{second}",
                                               kind="parse") from second
                else:
                    raise DialectReadError(f"官方解析失败：{message}", kind="parse") from first

            if verify:
                verified, error = verify_module(parsed.operation)
                module.diagnostics.append({"check": "verifier", "verified": verified,
                                           "error": error})
            module.module_attributes = {
                name: str(parsed.operation.attributes[name])
                for name in parsed.operation.attributes}
            lines = text.splitlines()
            text_locations = re.search(r"\bloc\s*\(", text) is None

            def walk(op, node_path, parent):
                name = op.operation.name
                dialect = name.split(".", 1)[0]
                line, ir_location = _location(op.operation)
                if not text_locations:
                    line = None
                snippet = ""
                if line is not None and 1 <= line <= len(lines):
                    snippet = lines[line - 1].strip()
                operands = tuple(str(value.type) for value in op.operation.operands)
                results = tuple(str(value.type) for value in op.operation.results)
                attributes = {key: str(op.operation.attributes[key])
                              for key in op.operation.attributes}
                module.ops.append(DialectOp(
                    name=name, dialect=dialect, line=line, file=str(path),
                    node_path=node_path, operands=operands, results=results,
                    types=operands + results, attributes=attributes,
                    region_count=len(op.operation.regions), text=snippet,
                    parent=parent, ir_location=ir_location,
                    implicit=line is None, source="official"))
                if name == "func.func" and "sym_name" in attributes:
                    module.symbol_defs.append(_split_symbol(attributes["sym_name"]))
                elif name == "func.call" and "callee" in attributes:
                    module.callees.append(_split_symbol(attributes["callee"]))
                for region_index, region in enumerate(op.operation.regions):
                    for block_index, block in enumerate(region.blocks):
                        for op_index, child in enumerate(block.operations):
                            walk(child, f"{node_path}.r{region_index}.b{block_index}.op{op_index}",
                                 name)

            walk(parsed, "module", None)
        finally:
            handle.detach()

    module.diagnostics.extend(records)
    module.nonstandard_dialects = sorted(
        {op.dialect for op in module.ops} - STANDARD_DIALECTS)
    return module

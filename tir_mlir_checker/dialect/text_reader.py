"""Stdlib-only MLIR text scanner.

It builds the same ``DialectModule`` fact model as the official reader, but
from printed assembly.  Everything it cannot know is recorded in ``limits``;
in particular implicit regions/terminators (buffer-form ``linalg.fill``) and
elided prefixes are text-invisible, and operand/result counts are approximate.
"""
from __future__ import annotations

import hashlib
import re

from .model import DialectModule, DialectOp
from .profiles import STANDARD_DIALECTS

OP_ALIASES = {"return": "func.return"}
TYPE_NAMES = ("memref", "tensor", "vector", "complex")
SCALAR_TYPES = frozenset({
    "bf16", "f16", "f32", "f64", "f80", "f128", "index",
    "i1", "i2", "i4", "i8", "i16", "i32", "i64",
    "u1", "u2", "u4", "u8", "u16", "u32", "u64",
})

MODULE_RE = re.compile(r"^\s*module\b")
OP_RE = re.compile(
    r"^\s*(?:(%[\w.$#-]+(?:\s*,\s*%[\w.$#-]+)*)\s*=\s*)?"
    r'(?:"([A-Za-z_][\w.$-]*)"|([A-Za-z_][\w]*\.[\w]+)|(return|yield))'
    r"(?=[\s({:<,]|$)"
)
SSA_RE = re.compile(r"%[\w.$#-]+")
FUNC_DEF_RE = re.compile(r"\bfunc\.func\s+(?:(?:private|public|nested)\s+)*@([\w.$-]+)")
FUNC_CALL_RE = re.compile(r"\bfunc\.call\s+@([\w.$-]+)")
MODULE_ATTR_RE = re.compile(r"^\s*module\s+attributes\b")


def _strip_comment(line):
    out, quoted = [], False
    i = 0
    while i < len(line):
        char = line[i]
        if char == '"':
            quoted = not quoted
        if not quoted and line.startswith("//", i):
            break
        out.append(char)
        i += 1
    return "".join(out)


def _brace_delta(code):
    opens = closes = 0
    quoted = False
    for char in code:
        if char == '"':
            quoted = not quoted
        elif not quoted and char == "{":
            opens += 1
        elif not quoted and char == "}":
            closes += 1
    return opens - closes


def _match_brace(code, start):
    quoted, depth = False, 0
    for index in range(start, len(code)):
        char = code[index]
        if char == '"':
            quoted = not quoted
        elif not quoted and char == "{":
            depth += 1
        elif not quoted and char == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _split_top_level(content):
    items, current, depth, quoted = [], [], 0, False
    for char in content:
        if char == '"':
            quoted = not quoted
        elif not quoted and char in "<([{":
            depth += 1
        elif not quoted and char in ">)]}":
            depth -= 1
        if not quoted and depth == 0 and char == ",":
            items.append("".join(current))
            current = []
            continue
        current.append(char)
    if "".join(current).strip():
        items.append("".join(current))
    return [item.strip() for item in items]


def _attributes(code, start):
    """Keys of balanced attribute dictionaries after ``start``."""
    pairs, index = {}, start
    while index < len(code):
        if code[index] == "{":
            end = _match_brace(code, index)
            if end is not None:
                for part in _split_top_level(code[index + 1:end]):
                    if "=" in part:
                        key, value = part.split("=", 1)
                        key = key.strip()
                        if re.fullmatch(r"[A-Za-z_][\w.$]*", key):
                            pairs[key] = value.strip()
                index = end + 1
                continue
        index += 1
    return pairs


def _mask_groups(code, start):
    """Blank out balanced attribute dictionaries so their types are not data dtypes."""
    chars = list(code)
    index = start
    while index < len(code):
        if code[index] == "{":
            end = _match_brace(code, index)
            if end is not None:
                for position in range(index, end + 1):
                    chars[position] = " "
                index = end + 1
                continue
        index += 1
    return "".join(chars)


def _balanced_types(code):
    found = []
    for match in re.finditer(r"\b(?:%s)<" % "|".join(TYPE_NAMES), code):
        start = match.end() - 1
        depth, quoted = 0, False
        for index in range(start, len(code)):
            char = code[index]
            if char == '"':
                quoted = not quoted
            elif not quoted and char == "<":
                depth += 1
            elif not quoted and char == ">":
                depth -= 1
                if depth == 0:
                    found.append(code[match.start():index + 1])
                    break
    return found


def _scalar_types(code):
    found = []
    for match in re.finditer(r"(?::|\bto)\s*([A-Za-z_][\w]*)\b", code):
        token = match.group(1)
        if token in SCALAR_TYPES or token in TYPE_NAMES:
            found.append(token)
    return found


def _types(code):
    values = _balanced_types(code) + _scalar_types(code)
    seen, ordered = set(), []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)


def read_text(text, *, path="<text>", sha256=None):
    """Scan printed MLIR into a DialectModule (facts only, no verification)."""
    module = DialectModule(path=path, sha256=sha256 or hashlib.sha256(text.encode()).hexdigest(),
                           backend="text", parse_mode="text")
    module.limits.extend([
        "text 后端看不到隐式 region/terminator（如 buffer 形式 linalg.fill 内部的 linalg.yield）",
        "elided 操作名按别名表规范化（return→func.return）；其余未打印前缀的写法无法识别",
        "操作数/结果/region 计数与属性字典为同行近似解析；arity 判定仅在 official 后端启用",
        "跨行操作签名（如 func.func 的参数表折行）只按首行解析，region 层级可能不完整",
    ])
    depth = 0
    stack = []  # (op, depth_after_open)
    lines = text.splitlines()
    for lineno, raw in enumerate(lines, 1):
        code = _strip_comment(raw)
        stripped = code.strip()
        if not stripped:
            continue
        net = _brace_delta(code)

        # A pure closing line pops regions; the parent is only known by depth.
        if net < 0:
            depth += net
            while stack and stack[-1][1] > depth:
                stack.pop()

        if stripped.startswith("^") or stripped.startswith("#") or stripped.startswith("!"):
            depth += max(net, 0)
            continue

        parent = stack[-1][0] if stack else None
        snippet = stripped

        if MODULE_RE.match(code):
            op = DialectOp(name="builtin.module", dialect="builtin", line=lineno,
                           file=str(path), node_path="module", text=snippet,
                           source="text")
            if MODULE_ATTR_RE.match(code):
                module.module_attributes = _attributes(code, code.find("attributes"))
            module.ops.append(op)
            if net > 0:
                depth += net
                stack.append((op, depth))
            continue

        match = OP_RE.match(code)
        if not match:
            if net > 0:
                depth += net
            continue

        name = match.group(2) or match.group(3) or match.group(4)
        name = OP_ALIASES.get(name, name)
        dialect = name.split(".", 1)[0]
        index = len(module.ops)
        node_path = f"{parent.node_path}/op{index}" if parent and parent.node_path else f"op{index}"
        operands_text = code[match.end():]
        operands = tuple(dict.fromkeys(SSA_RE.findall(operands_text)))
        results = tuple(SSA_RE.findall(match.group(1) or ""))
        op = DialectOp(
            name=name, dialect=dialect, line=lineno, file=str(path),
            node_path=node_path, operands=operands, results=results,
            types=_types(_mask_groups(code, match.end())),
            attributes=_attributes(code, match.end()),
            region_count=net if net > 0 else 0, text=snippet,
            parent=parent.name if parent else None, source="text",
        )
        module.ops.append(op)
        if name == "func.func":
            module.symbol_defs.extend(FUNC_DEF_RE.findall(code))
        elif name == "func.call":
            module.callees.extend(FUNC_CALL_RE.findall(code))
        if net > 0:
            depth += net
            stack.append((op, depth))

    module.nonstandard_dialects = sorted(
        {op.dialect for op in module.ops} - STANDARD_DIALECTS)
    return module

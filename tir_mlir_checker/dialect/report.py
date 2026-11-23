"""Dialect-layer result assembly and optional evidence output."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .. import __version__
from ..diagnostics import agent_diagnostics, diagnostic_text

EXIT_CODES = {"ok": 0, "mismatch": 1, "input_error": 2, "tool_error": 3, "unsupported": 4}
SCOPE = ("当前模块的方言/操作结构与 profile 条款；不执行数值，"
         "不证明跨阶段语义等价或机器码正确。")


def build_result(module, profile, findings, *, stage=None, mode="check"):
    counts = Counter(item["status"] for item in findings)
    mismatch = counts["mismatch"]
    unsupported = counts["unsupported"]
    uncertain = counts["uncertain"]
    tool_status = "mismatch" if mismatch else ("unsupported" if unsupported else "ok")
    conclusion = ("bounded_checks_failed" if mismatch else
                  "unsupported" if unsupported else
                  "bounded_checks_passed_with_pending" if uncertain else
                  "bounded_checks_passed")
    result = {
        "schema_version": 1,
        "layer": "dialect",
        "mode": mode,
        "tool_version": __version__,
        "tool_status": tool_status,
        "exit_code": EXIT_CODES[tool_status],
        "dialect_status": {"ok": "passed", "mismatch": "mismatch",
                           "unsupported": "unsupported"}[tool_status],
        "semantic_conclusion": conclusion,
        "profile": profile.public(),
        "profile_notes": list(profile.notes),
        "backend": module.backend,
        "module": module.public(),
        "stage": stage,
        "summary": {
            "mismatch": mismatch, "uncertain": uncertain, "unsupported": unsupported,
            "findings": len(findings), "ops": len(module.ops),
            "dialects": len(module.dialects()),
        },
        "findings": findings,
        "scope": SCOPE,
        "next_step": ("按 rule/location 回查原始行或 node_path；uncertain 与 pending "
                      "不是通过，也不是错误。证据位置以 artifacts 为准。"),
    }
    result["agent_diagnostics"] = agent_diagnostics(result)
    result["diagnostics_text"] = diagnostic_text(result["agent_diagnostics"])
    return result


def _finding_lines(result):
    lines = []
    for item in result["findings"]:
        loc = (item.get("locations") or [{}])[0]
        site = loc.get("file") or loc.get("stage") or "<module>"
        if loc.get("line"):
            site += f":{loc['line']}"
        elif loc.get("node_path"):
            site += f" ({loc['node_path']})"
        lines.append(f"- `[{item['rule_id']}; {item['status']}]` {item['message']} — {site}")
        if loc.get("excerpt"):
            lines.append(f"  ```text\n  {loc['excerpt']}\n  ```")
    return lines or ["- 无。"]


def write_outputs(result, out_dir):
    directory = Path(out_dir).expanduser()
    if directory.exists():
        raise FileExistsError(f"输出目录必须不存在：{directory}")
    directory.mkdir(parents=True)
    (directory / "dialect-result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pending = next((item for item in result["findings"]
                    if item["rule_id"] == "DIA-PROFILE-PENDING"), None)
    markdown = [
        "# Dialect 层检查记录",
        "",
        "## 1. 检查结论",
        "",
        f"- 状态：`{result['tool_status']}`（exit {result['exit_code']}）",
        f"- 结论：`{result['semantic_conclusion']}`",
        f"- 契约：`{result['profile']['name']}:{result['profile']['version']}`"
        f"（{result['profile']['status']}）",
        f"- 摘要：mismatch={result['summary']['mismatch']}，"
        f"uncertain={result['summary']['uncertain']}，"
        f"unsupported={result['summary']['unsupported']}，"
        f"ops={result['summary']['ops']}，dialects={result['summary']['dialects']}",
        "",
        "## 2. 怎么查的与测试范围",
        "",
        f"- 模块：`{result['module']['path']}`（sha256 `{result['module']['sha256'][:16]}…`）",
        f"- 后端：`{result['backend']}`（parse_mode `{result['module']['parse_mode']}`）"
        + (f"，回退原因：{result['module']['fallback_reason']}"
           if result["module"].get("fallback_reason") else ""),
        f"- 阶段：`{result['stage'] or '未指定'}`",
        f"- 方言：{', '.join(result['module']['dialects']) or '无'}",
        f"- 入口符号：{', '.join(result['module']['entry_symbols']) or '无'}",
        "- 数据流：读取方言/操作/属性/类型 → 按 profile 条款机械核验；"
        "官方后端可另跑 verifier。",
        "- 边界：不执行数值，不证明语义等价。",
    ]
    markdown += [f"- 后端限制：{limit}" for limit in result["module"].get("limits", [])]
    markdown += ["", "## 3. 问题定位与判断", ""]
    markdown += _finding_lines(result)
    if pending:
        markdown += ["", "### 未确认条款（不是错误）", ""]
        markdown += [f"- {item}" for item in pending["details"].get("pending", [])]
    markdown += ["", f"> {result['scope']}", ""]
    (directory / "dialect-report.md").write_text("\n".join(markdown), encoding="utf-8")
    return directory

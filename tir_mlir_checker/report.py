"""Two output files: human-readable report and a structured evidence ZIP."""
from __future__ import annotations

import json
import base64
import zipfile
from pathlib import Path

from .trace_io import InputError


def _escape(value):
    return str(value).replace("|", "\\|").replace("\n", "<br>")




# Keep machine diagnostics intact; the visible record has three sections.
def render_report(result):
    from .diagnostics import agent_diagnostics
    from .naming import artifact_names, identify_operator
    identity = result.get('operator') or identify_operator(result)
    name = identity['name']
    names = artifact_names(name)
    dyn = result['dynamic']
    tests = dyn.get('tests', [])
    passed = sum(t.get('status') == 'match' for t in tests)
    errors = [t.get('max_finite_abs_error') for t in tests]
    errors = [e for e in errors if e is not None]
    functions = ', '.join(dict.fromkeys(n for items in identity['ir_functions'].values() for n in items)) or '未识别'
    cases = dyn.get('cases', [])
    case_summary = '; '.join(f"{c['name']}: " + ', '.join(
        f"arg{i}={a['shape']}/{a['dtype']}" for i, a in enumerate(c['arguments'])) for c in cases)
    if result['tool_status'] == 'mismatch':
        conclusion = '发现具体违规或数值反例。'
    elif result['tool_status'] != 'ok':
        conclusion = '本次检查未完整完成，具体诊断见检查证据。'
    elif not tests:
        conclusion = '静态检查已完成；数值对照未执行。'
    else:
        conclusion = f'本次数值对照 {passed}/{len(tests)} 次通过。'
    lines = [f'# {name} 自动检查记录', '', '## 1. 检查结论', '', conclusion,
             f"原始状态：`{result['tool_status']}`；`{result['semantic_conclusion']}`；汇总：{_escape(result['summary'])}。",
             '', '## 2. 测试范围与结果', '', '| 项目 | 本次结果 |', '| --- | --- |',
             f"| 算子 / 真实IR函数 | {_escape(name)} / {_escape(functions)} |",
             f"| 版本 / 批次 | {_escape(result['tool_version'])} / {_escape(result['input'].get('run_id') or '未记录')} |",
             f"| 输入范围 | {_escape(case_summary or '未提供动态用例；形状和类型见证据facts.json')} |",
             f"| 受限解释 / 数值对照 | {str(passed)+'/'+str(len(tests))+' 次通过' if tests else '未执行'}；最大有限绝对误差：{max(errors) if errors else '未取得'} |",
             f"| 正式 MLIR verifier | {_escape(result.get('mlir_verification', {}).get('status', '未执行'))} |",
             f"| 官方 MLIR Python 解析 / 验证 | {_escape(result.get('mlir_python_verification', {}).get('status', '未执行'))}；对象模拟与verifier分别记录 |",
             f"| 官方 TIR 读取 / SSA与结构检查 | {_escape(result.get('tir_verification', {}).get('status', '未执行'))}；与受限节点模拟分别记录 |",
             '| 本次真机执行 | 未执行 |',
             '', '## 3. 问题与 Agent 判断', '',
             'Python诊断按规则、状态和原始位置合并。',
             '', '| 编号 / 原状态 | 发生了什么 | 关键位置 |', '| --- | --- | --- |']
    groups = {}
    for item in agent_diagnostics(result):
        loc = item['location'] or {}
        notes = [note['location'] for note in item['notes']]
        key = (item['rule_id'], item['status'], item['message'], loc.get('stage'),
               loc.get('file'), loc.get('line'),
               tuple((n.get('stage'), n.get('file'), n.get('line')) for n in notes))
        entry = groups.setdefault(key, {'item': item, 'ids': []})
        entry['ids'].append(item['id'])
    for entry in list(groups.values())[:15]:
        item, ids = entry['item'], entry['ids']
        loc = item['location'] or {}
        def position(p):
            return f"{p.get('stage') or Path(p.get('file') or '未知').name}:{p.get('line') or p.get('node_path') or '?'}"
        pos = ' → '.join([position(loc)] + [position(n['location']) for n in item['notes']])
        lines.append('| ' + ' | '.join(_escape(v) for v in
            (','.join(ids) + ' / ' + item['status'],
             item['message'] + '（' + item['rule_id'] + '）', pos)) + ' |')
    if not groups:
        lines.append('| — | 本次未发现具体自动诊断 | — |')
    if len(groups) > 15:
        lines.append('仅展示前15类；完整诊断及关联用例见证据。')
    supplement = result.get('agent_translation')
    if supplement is not None:
        tests = supplement.get('tests', [])
        passed = sum(t['status'] == 'match' for t in tests)
        marker = '| 本次真机执行 |'
        index = next(i for i, line in enumerate(lines) if line.startswith(marker))
        lines.insert(index, f"| Agent翻译后样本验证（单列） | {_escape(supplement['status'])}；{passed}/{len(tests)}次比较通过；输入绑定：{_escape(supplement.get('input_binding'))} |")
        lines += ['', '翻译代码、IR映射、输入与输出见证据中 `agent_translation/`。']
    lines += ['', f"完整诊断、原始IR与数组：[检查证据](<{names['evidence']}>)；规则和触发条件见 `result.json`、`facts.json`、`diagnostics.txt`。", '']
    return '\n'.join(lines)


def write_report(result, out_dir):
    from .diagnostics import agent_diagnostics, diagnostic_text
    from .naming import artifact_names, identify_operator
    out = Path(out_dir).expanduser().resolve()
    root = Path(result["input"]["root"])
    if out == root or out.is_relative_to(root):
        raise InputError("报告目录不得位于输入 Trace 目录中；请使用独立临时目录")
    # run has just created out/trace; allow only that exact directory, no outputs.
    if out.exists():
        if not (out == root.parent and root.name == 'trace' and set(out.iterdir()) == {root}):
            raise InputError(f"报告目录已存在，请选择新的目录以免覆盖或混用运行：{out}")
    result['operator'] = result.get('operator') or identify_operator(result)
    names = artifact_names(result['operator']['name'])
    result['report_naming'] = {'final_report': names['final_report'],
                              'final_title': names['final_report'][:-3]}
    out.mkdir(parents=True, exist_ok=True)
    paths = {key: str(out / names[key]) for key in ('report', 'evidence')}
    result["artifacts"] = paths
    result["agent_diagnostics"] = agent_diagnostics(result)
    # Split large facts from the result protocol. Python API still returns both.
    payload = {k: v for k, v in result.items() if k not in {"facts","_evidence"}}
    payload["facts_member"] = "facts.json"
    with zipfile.ZipFile(paths["evidence"], "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("result.json", json.dumps(payload, ensure_ascii=False, indent=2))
        archive.writestr("facts.json", json.dumps(result["facts"], ensure_ascii=False, indent=2))
        archive.writestr("diagnostics.txt", diagnostic_text(result["agent_diagnostics"]))
        for name,encoded in result.get('_evidence',{}).items():
            archive.writestr(name,base64.b64decode(encoded))
    Path(paths["report"]).write_text(render_report(result), encoding="utf-8")

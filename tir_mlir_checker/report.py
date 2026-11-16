"""Two output files: human-readable report and a structured evidence ZIP."""
from __future__ import annotations

import json
import base64
import zipfile
from pathlib import Path

from .trace_io import InputError


def _escape(value):
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _code(value):
    return '`' + _escape(value).replace('`', '&#96;') + '`'


def _execution_summary(dynamic):
    # A multi-output case is one execution, with several output comparisons.
    tests = dynamic.get('tests', [])
    runs = {}
    for test in tests:
        runs.setdefault((test.get('stage'), test.get('case')), []).append(test)
    passed = sum(all(t.get('status') == 'match' for t in items) for items in runs.values())
    failed = sum(any(t.get('status') == 'mismatch' for t in items) for items in runs.values())
    policies = [t.get('policy', {}) for t in tests]
    modes = {p.get('mode') for p in policies}
    comparison = '逐位一致' if modes == {'bitwise'} else '容差一致' if modes == {'tolerance'} else '按各用例契约一致'
    errors = [t['max_finite_abs_error'] for t in tests if t.get('max_finite_abs_error') is not None]
    stages = list(dict.fromkeys(t.get('stage') for t in tests if t.get('stage')))
    cases = list(dict.fromkeys(t.get('case') for t in tests if t.get('case')))
    parts = [f'{len(cases)} 组输入 × {len(stages)} 阶段',
             f"累计 {sum(t.get('elements', 0) for t in tests)} 个输出元素对照",
             f"最大有限绝对误差 {max(errors) if errors else '未取得'}"]
    text = (f'**{passed} / {len(runs)} 次{comparison}**（' + '，'.join(parts) + '）') if runs else '未执行'
    if failed:
        text += f'；{failed} 次模拟失败'
    missing = len(runs) - passed - failed
    if missing:
        text += f'；{missing} 次未完成'
    return dict(tests=tests, runs=runs, passed=passed, failed=failed, stages=stages,
                cases=cases, comparison=comparison, text=text)


def _tensor_signatures(cases):
    groups = {}
    for case in cases:
        outputs, inouts = case.get('outputs', []), case.get('inouts', [])
        signature = ', '.join(
            f"arg{i}{'(inout)' if i in inouts else '(out)' if i in outputs else ''}: "
            f"{arg['dtype']}[{','.join(map(str, arg['shape']))}]"
            for i, arg in enumerate(case.get('arguments', [])))
        groups.setdefault(signature, []).append(case['name'])
    if len(groups) == 1:
        return _code(next(iter(groups)))
    return '<br>'.join(_code(signature) + '（' + _escape(', '.join(names)) + '）'
                       for signature, names in groups.items()) or '未提供动态用例；张量信息见 facts.json'


def _official_summary(result):
    labels = [('tir_verification', 'TIR 00–02 读取 / SSA / 结构'),
              ('mlir_python_verification', 'MLIR 03–04 解析 / verifier'),
              ('mlir_verification', 'mlir-opt 03–04')]
    states = {'passed': '通过', 'not_requested': '未执行', 'unsupported': '未完成（依赖或格式不支持）',
              'error': '未通过（原因见诊断）', 'invalid_ir': 'IR 结构不合法'}
    return '<br>'.join(f"{label}：{states.get(result.get(key, {}).get('status', 'not_requested'), '未完成')}"
                       for key, label in labels)


def _review_items(result, diagnostics):
    items = []
    uncertain = [item for item in diagnostics if item['status'] == 'uncertain']
    for check in result.get('checks', []):
        if check.get('status') == 'uncertain':
            # The same observation can be both a check and a located finding.
            # Prefer the finding, which has a rule name and an IR position.
            if any(e.get('stage') == (item['location'] or {}).get('stage')
                   and (item['location'] or {}).get('line') in e.get('lines', [])
                   for e in check.get('evidence', []) for item in uncertain):
                continue
            items.append((check['id'], check.get('scope', check.get('detail', '需回查原始 IR'))))
    seen = set()
    for item in uncertain:
        loc = item['location'] or {}
        key = (item['rule_id'], loc.get('stage'), loc.get('file'), loc.get('line'), loc.get('node_path'))
        if key not in seen:
            seen.add(key)
            site = f"（{loc['stage']}:{loc.get('line') or loc.get('node_path') or '?'}）" if loc.get('stage') else ''
            items.append((item['rule_id'] + site, item['message']))
    return items


def _diagnostic_groups(diagnostics):
    groups = {}
    for item in diagnostics:
        if item['status'] == 'uncertain':
            continue
        loc = item['location'] or {}
        key = (item['rule_id'], item['status'], loc.get('stage'), loc.get('file'),
               loc.get('line'), loc.get('node_path'), loc.get('ir_location'))
        group = groups.setdefault(key, [])
        group.append(item)
    return list(groups.values())


def _problem_card(items):
    first = next((i for i in items if i['details'].get('first_difference')), items[0])
    loc, details = first['location'] or {}, first['details']
    file = Path(loc.get('file') or loc.get('stage') or '未知位置').name
    position = f"{file}:{loc.get('line') or loc.get('node_path') or '?'}"
    cases = list(dict.fromkeys(i['details']['case'] for i in items if i['details'].get('case')))
    label = '❌' if first['status'] in {'mismatch', 'input_error'} else '⚠️'
    lines = [f"> {label} **{_escape(first['message'])} [{_escape(first['rule_id'])}]**",
             f"> - **报错阶段与位置（症状点）：** {_code(loc.get('stage') or '未定位阶段')}，{_code(position)}"]
    if loc.get('excerpt'):
        lines.append(f"> - **对应代码：** {_code(loc['excerpt'].strip())}")
    if cases:
        lines.append(f"> - **影响用例（{len(cases)} 组）：** " + ', '.join(_code(c) for c in cases))
    difference = details.get('first_difference')
    if difference:
        index = ','.join(map(str, difference.get('index', [])))
        output = f"arg{details.get('output', '?')}(out)[{index}]"
        actual, expected = difference.get('actual', {}), difference.get('expected', {})
        lines.append(f"> - **首个反例：** {_code(details.get('case', '未记录'))} 的 {_code(output)}："
                     f"实际 {_code(actual.get('value', actual))}，独立参考 {_code(expected.get('value', expected))}")
        if actual.get('bits') is not None and expected.get('bits') is not None:
            lines.append(f"> - **原始字节（actual / expected）：** {_code(actual['bits'])} / {_code(expected['bits'])}")
    elif details.get('index') is not None:
        lines.append(f"> - **触发下标：** {_code(details['index'])}")
    if first['notes']:
        sites = []
        for note in first['notes']:
            p = note['location']
            sites.append(f"{Path(p.get('file') or p.get('stage') or '未知').name}:{p.get('line') or p.get('node_path') or '?'}")
        lines.append('> - **相关位置：** ' + ', '.join(_code(s) for s in sites))
    lines.append('> **Agent 根因分析：** [待 Agent 沿 IR 数据流回溯确认]')
    return lines


# Rendering is read-only: machine statuses, counts and evidence remain intact.
def render_report(result):
    from .diagnostics import agent_diagnostics
    from .naming import artifact_names, identify_operator
    identity = result.get('operator') or identify_operator(result)
    name = identity['name']
    names = artifact_names(name)
    dynamic = result.get('dynamic', {})
    execution = _execution_summary(dynamic)
    diagnostics = agent_diagnostics(result)
    reviews = _review_items(result, diagnostics)
    problems = _diagnostic_groups(diagnostics)
    invalid_stages = [run['stage'] for key in ('tir_verification', 'mlir_python_verification')
                      for run in result.get(key, {}).get('runs', []) if run.get('status') == 'invalid_ir']
    failures = [t for t in execution['tests'] if t.get('status') == 'mismatch']
    failed_stages = list(dict.fromkeys([t['stage'] for t in failures] + invalid_stages +
                         [i['location']['stage'] for i in diagnostics if i['status'] == 'mismatch'
                          and i['location'] and i['location'].get('stage')]))
    has_error = result['tool_status'] == 'mismatch' or bool(failures or invalid_stages)
    complete = (result['tool_status'] == 'ok' and dynamic.get('status') == 'passed'
                and dynamic.get('complete', True) and execution['runs']
                and execution['passed'] == len(execution['runs']))
    if has_error:
        stage = ', '.join(failed_stages) or '检查过程中'
        verdict = f"❌ **失败**（在 {stage} 发现错误，共 {execution['failed']} 次模拟失败）"
    elif complete:
        verdict = '✅ **通过**（未发现语义错误）'
    elif result['tool_status'] == 'ok' and not dynamic.get('requested_dynamic', dynamic.get('executed')):
        verdict = '⚪ **静态检查完成**（未执行数值模拟）'
    else:
        verdict = '⚠️ **检查未完成**（具体原因见问题定位）'
    if execution['runs']:
        summary = (f"{len(execution['stages'])} 个编译阶段、{len(execution['cases'])} 组输入，"
                   f"共 {len(execution['runs'])} 次模拟，{execution['passed']} 次{execution['comparison']}")
        if execution['failed']:
            summary += f"，{execution['failed']} 次失败"
    else:
        summary = '本轮执行文件与结构检查，数值模拟未执行'
    if reviews:
        summary += f'；另有 {len(reviews)} 项需 Agent 结合原始 IR 复核'
    functions = ', '.join(dict.fromkeys(n for ns in identity['ir_functions'].values() for n in ns)) or '未识别'
    backends = {v.get('backend') for v in dynamic.get('stage_support', {}).values()}
    model = '内置对象解释器（TIR 节点 / MLIR SSA 与 regions）' if backends and backends <= {'tvm_tir_nodes', 'mlir_operation_objects'} else '内置 IR 解释器（实际后端见下表）'
    backend_names = {'tvm_tir_nodes': 'TIR 对象节点', 'mlir_operation_objects': 'MLIR 操作对象',
                     'tir_ast': 'TIR AST', 'mlir_text': 'MLIR 文本'}
    actual_backends = ', '.join(backend_names.get(b, str(b)) for b in sorted(backends - {None})) or '未执行'
    cases = dynamic.get('cases', [])
    policies = list(dict.fromkeys(
        f"逐位比较" if t.get('policy', {}).get('mode') == 'bitwise' else
        f"容差比较（rtol={t['policy'].get('rtol')}, atol={t['policy'].get('atol')}）" if t.get('policy', {}).get('mode') == 'tolerance'
        else '比较策略见 result.json' for t in execution['tests']))
    lines = [f'# {name} 自动检查记录', '', '## 1. 检查结论', '', verdict, '', summary + '。',
             '', '## 2. 怎么查的与测试范围', '', '**检查机制：**', '',
             '1. **读取编译快照：** 读取五阶段 IR（`00_tir_initial` → `04_mlir_final`）并核对清单指纹；新快照由 `export` 从真实编译流程截取。',
             '2. **结构合法性：** 按配置调用 TVM 的 SSA/结构检查、MLIR 官方解析/verifier 与 `mlir-opt`，各项实际状态见下表。',
             f'3. **独立数值模拟：** 使用{model}，对各阶段独立灌入相同输入并与独立参考比对；未执行算子真机机器码或硬件跑分。',
             '', '| 维度 | 实际配置与结果 |', '| --- | --- |',
             f"| 算子 / 函数 / 批次 | {_code(name)} / {_code(functions)} / {_code(result['input'].get('run_id') or '未记录')}（Checker {_code(result['tool_version'])}） |",
             f"| 张量签名（已去重） | {_tensor_signatures(cases)} |",
             f"| 测试用例（共 {len(cases)} 组） | {', '.join(_code(c['name']) for c in cases) or '未提供'} |",
             f"| 官方结构验证 | {_official_summary(result)} |",
             f"| 动态数值模拟 | {execution['text']} |",
             f"| 解释后端 / 比较策略 | {_escape(actual_backends)} / {'；'.join(policies) or '未执行'} |"]
    supplement = result.get('agent_translation')
    if supplement is not None:
        extra = _execution_summary(supplement)
        lines.append(f"| Agent 翻译后样本验证（单列） | {_escape(supplement['status'])}；{extra['text']}；输入绑定：{_escape(supplement.get('input_binding'))} |")
    lines += ['', '## 3. 问题定位与 Agent 复核', '']
    if not problems:
        messages = result.get('diagnostics', [])
        lines.append('**自动报错诊断：** ' + ('；'.join(_escape(m) for m in messages) if messages else '无。'))
    for group in problems:
        lines += _problem_card(group) + ['']
    if reviews:
        lines += ['', f'**待 Agent 复核项（{len(reviews)} 项）：**', '',
                  '| 规则 / 核查内容 | Agent 复核 |', '| --- | --- |']
        for rule, scope in reviews:
            lines.append(f'| {_escape(rule)}：{_escape(scope)} | [待 Agent 复核] |')
    else:
        lines.append('**待 Agent 复核项：** 无。')
    lines += ['', f"> 完整诊断日志、规则字典、原始 IR 与数组见[检查证据](<{names['evidence']}>)（`result.json`、`facts.json`、`diagnostics.txt`）。", '']
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

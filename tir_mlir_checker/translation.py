"""Execute explicit Agent translations as supplementary, unverified IR models.

No model API, automatic translation, symbolic analysis or hardware claim is made.
This module is loaded in the existing timeout-controlled worker subprocess.
"""
from __future__ import annotations

import base64
import hashlib
import io
import itertools
import runpy
from pathlib import Path


def execute_translation(request):
    import numpy as np
    from .cases import compare, validate_case, cases_fingerprint
    from .diagnostics import finding
    from .execution import Unsupported
    from .trace_io import load_trace

    group = load_trace(request['trace'])
    source = Path(request['translation']).read_bytes()
    namespace = runpy.run_path(request['translation'], run_name='checker_agent_translation')
    specs = namespace.get('TRANSLATIONS')
    if not isinstance(specs, dict) or not specs:
        raise Unsupported('Agent翻译必须声明非空TRANSLATIONS：阶段→sha256/entry/line_map/assumptions')
    for stage, spec in specs.items():
        if stage not in request['stages'] or not isinstance(spec, dict):
            raise Unsupported('Agent翻译包含未知或未请求的阶段')
        actual = hashlib.sha256(group.text[stage].encode()).hexdigest()
        if spec.get('sha256') != actual:
            raise Unsupported(f'{stage}：Agent翻译绑定的IR哈希与当前快照不同')
        if not callable(namespace.get(spec.get('entry', ''))):
            raise Unsupported(f'{stage}：翻译entry不是可调用函数')
        assumptions = spec.get('assumptions')
        mapping = spec.get('line_map')
        if not isinstance(assumptions, list) or not assumptions or not all(isinstance(a, str) and a.strip() for a in assumptions):
            raise Unsupported(f'{stage}：必须明确列出翻译适用前提')
        if not isinstance(mapping, list) or not mapping:
            raise Unsupported(f'{stage}：必须提供原IR与Python行号映射')
        for item in mapping:
            for key, limit in [('ir', len(group.text[stage].splitlines())), ('python', len(source.splitlines()))]:
                value = item.get(key) if isinstance(item, dict) else None
                if (not isinstance(value, list) or len(value) != 2 or
                        any(type(v) is not int for v in value) or not 1 <= value[0] <= value[1] <= limit):
                    raise Unsupported(f'{stage}：{key}行号映射越界或格式错误')

    adapter = runpy.run_path(request['case'], run_name='checker_case_adapter')
    if not callable(adapter.get('make_cases')):
        raise Unsupported('测试适配必须定义make_cases(seed)')
    cases = list(itertools.islice(adapter['make_cases'](request['seed']), 33))
    if not 1 <= len(cases) <= 32:
        raise Unsupported('每次要求1–32个测试用例')
    for case in cases:
        validate_case(case)
    if len({c.name for c in cases}) != len(cases):
        raise Unsupported('用例名称重复')
    result = {'status': 'passed', 'backend': 'agent_translation', 'executed': False,
              'cases_fingerprint': cases_fingerprint(cases),
              'translation_sha256': hashlib.sha256(source).hexdigest(),
              'translations': specs, 'tests': [], 'findings': [],
              'case_count': len(cases), 'seed': request['seed'],
              'fidelity': 'unverified_agent_translation',
              'scope': '只验证生成的Python在显式样本上的输出；IR翻译忠实性待Agent回查。无符号、内存覆盖或并行证明。'}
    arrays = {}

    def save(name, a):
        result.setdefault('array_dtypes', {})[name] = str(a.dtype)
        arrays[name] = np.ascontiguousarray(a).view(np.uint8).reshape(a.shape + (a.dtype.itemsize,))

    for i, case in enumerate(cases):
        for pos, a in enumerate(case.arguments):
            if pos not in case.outputs or pos in case.inouts:
                save(f'case{i}_arg{pos}', a)
        for pos, expected in zip(case.outputs, case.expected):
            save(f'case{i}_expected{pos}', expected)
        for stage, spec in specs.items():
            arguments = tuple(np.array(a, copy=True) if pos not in case.outputs or pos in case.inouts
                              else np.empty_like(a) for pos, a in enumerate(case.arguments))
            readonly = {pos: a.tobytes() for pos, a in enumerate(arguments)
                        if pos not in case.outputs and pos not in case.inouts}
            for pos in readonly:
                arguments[pos].setflags(write=False)
            record = {'case': case.name, 'stage': stage, 'backend': 'agent_translation'}
            try:
                # Expected answers are never passed to generated functions.
                outputs = namespace[spec['entry']](arguments, dict(case.symbols))
                result['executed'] = True
                if (not isinstance(outputs, dict) or set(outputs) != set(case.outputs) or
                        any(type(pos) is not int for pos in outputs) or
                        any(not isinstance(a, np.ndarray) for a in outputs.values())):
                    raise Unsupported('翻译函数必须返回 {输出参数位置: ndarray}，且输出位置与用例完全一致')
                if any(arguments[pos].tobytes() != old for pos, old in readonly.items()):
                    raise Unsupported('生成代码修改了只读输入；需检查翻译实现')
                for pos, expected in zip(case.outputs, case.expected):
                    compared = compare(outputs[pos], expected, case.output_comparisons.get(pos, case.comparison))
                    result['tests'].append({**record, 'output': pos,
                                            'status': 'match' if compared['match'] else 'mismatch', **compared})
                    save(f'case{i}_{stage}_out{pos}', outputs[pos])
                    if not compared['match']:
                        result['findings'].append(finding('AGENT-TRANSLATION-DIFFERENCE', 'uncertain',
                            'agent_translation', 'Agent生成代码与独立参考不同；需先区分翻译错误与IR问题',
                            stage=stage, line=spec['line_map'][0]['ir'][0],
                            details={**record, 'output': pos, **compared, 'line_map': spec['line_map']},
                            scope='当前样本上的生成代码差异，不自动认定原IR或编译器错误。'))
            except Exception as exc:
                result['tests'].append({**record, 'status': 'unsupported' if isinstance(exc, Unsupported) else 'error',
                                        'reason': f'{type(exc).__name__}: {exc}'})
                result['findings'].append(finding('AGENT-TRANSLATION-EXECUTION', 'uncertain',
                    'agent_translation', f'{type(exc).__name__}: {exc}', stage=stage,
                    line=spec['line_map'][0]['ir'][0]))
    statuses = {t['status'] for t in result['tests']}
    result['status'] = next((s for s in ['error', 'mismatch', 'unsupported'] if s in statuses), 'passed')
    result['semantic_conclusion'] = ('translated_samples_match_ir_fidelity_unverified'
                                     if result['status'] == 'passed' else 'translation_requires_review')
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    result['_evidence'] = {'arrays.npz': base64.b64encode(stream.getvalue()).decode()}
    return result

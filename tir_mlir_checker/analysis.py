"""Operator-independent observations; deliberately limited semantic claims."""
from __future__ import annotations

import ast
import datetime
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from . import __version__
from .runtime import request_dynamic
from .trace_io import InputError, STAGES, load_trace
from .diagnostics import finding, locate
from .rules import analyze_rules

EXIT_CODES = {"ok": 0, "mismatch": 1, "input_error": 2, "tool_error": 3, "unsupported": 4}
SSA = r"%[\w.$-]+"
OP = re.compile(r'^\s*(?:%[\w.$,:# -]+\s*=\s*)?([A-Za-z_][\w]*\.[\w]+)\b')
HEADER = re.compile(r'\bfunc\.func\s+(?:(?:public|private|nested)\s+)?@([\w.$-]+)\s*\(([^()]*)\)')
MEMREF = re.compile(rf'({SSA})\s*:\s*memref<((?:(?:\d+|\?)x)*)([\w]+)>\s*$')
DTYPES = {"bfloat16": "bf16", "float16": "f16", "float32": "f32", "float64": "f64",
          "bool": "i1", **{f"{sign}int{bits}": f"i{bits}"
                           for sign in ("", "u") for bits in (8, 16, 32, 64)}}


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _call_name(node.value) + "." + node.attr
    return ""


def tir_facts(source):
    """AST observations only; Python is never imported/evaluated as TVMScript."""
    result = {"parser": "python_ast_observer", "formal_ir_verified": False,
              "signature": None, "functions": [], "events": []}
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        result.update(parse_status="unsupported", diagnostic=str(exc))
        return result
    result["parse_status"] = "observed"
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                 and any(_call_name(d) == "T.prim_func" for d in node.decorator_list)]
    result["functions"] = [{"name": n.name, "line": n.lineno} for n in functions]
    if len(functions) != 1:
        result["diagnostic"] = "目前仅观察单个 @T.prim_func 的参数，不猜测多函数映射。"
        return result
    function = functions[0]
    buffers = {}
    for node in function.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        call = node.value
        if not (isinstance(call, ast.Call) and _call_name(call.func) == "T.match_buffer"
                and isinstance(node.targets[0], ast.Name) and len(call.args) >= 2
                and isinstance(call.args[0], ast.Name)
                and isinstance(call.args[1], (ast.Tuple, ast.List))):
            continue
        dtype = next((k.value for k in call.keywords if k.arg == "dtype"), ast.Constant("float32"))
        if len(call.args) >= 3:
            dtype = call.args[2]
        if not isinstance(dtype, ast.Constant) or not isinstance(dtype.value, str):
            continue
        dims = [ast.unparse(d) for d in call.args[1].elts]
        storage_dtype = DTYPES.get(dtype.value)
        if storage_dtype is None:
            continue
        strides = next((ast.unparse(k.value) for k in call.keywords if k.arg == "strides"), None)
        buffers[call.args[0].id] = {
            "name": node.targets[0].id, "shape": dims,
            "dtype": dtype.value, "storage_dtype": storage_dtype,
            "strides": strides, "line": node.lineno,
            "dynamic_axes": [i for i, d in enumerate(dims) if not d.isdigit()],
        }
    params = function.args.posonlyargs + function.args.args
    if (not function.args.vararg and not function.args.kwarg and not function.args.kwonlyargs
            and all(p.arg in buffers for p in params)):
        result["signature"] = [dict(position=i, parameter=p.arg, **buffers[p.arg])
                               for i, p in enumerate(params)]
    for node in ast.walk(function):
        event = None
        if isinstance(node, ast.For):
            event = {"kind": "loop", "target": ast.unparse(node.target), "iterator": ast.unparse(node.iter)}
        elif isinstance(node, ast.If):
            event = {"kind": "guard", "condition": ast.unparse(node.test)}
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Subscript) for t in node.targets):
            event = {"kind": "buffer_assignment", "targets": [ast.unparse(t) for t in node.targets],
                     "value": ast.unparse(node.value)}
        elif isinstance(node, ast.Call) and _call_name(node.func).startswith("T."):
            event = {"kind": "call", "name": _call_name(node.func), "expression": ast.unparse(node)}
        if event is not None:
            result["events"].append(dict(line=node.lineno, **event))
    result["events"].sort(key=lambda e: e["line"])
    return result


def mlir_facts(source):
    """Line-based index, not an MLIR parser or a def-use/alias analysis."""
    # Mask comments/strings while keeping length and newlines for source positions.
    masked = re.sub(r'"(?:\\.|[^"\\])*"|//[^\n]*',
                    lambda m: "".join("\n" if c == "\n" else " " for c in m[0]), source)
    result = {"parser": "mlir_text_observer", "formal_ir_verified": False,
              "parse_status": "observed", "signature": None, "functions": [], "events": []}
    headers = list(HEADER.finditer(masked))
    for header in headers:
        result["functions"].append({"name": header[1], "line": masked.count("\n", 0, header.start()) + 1})
    if len(headers) == 1:
        header = headers[0]
        args, offset = [], header.start(2)
        parts = header[2].split(",") if header[2].strip() else []
        for part in parts:
            match = MEMREF.fullmatch(part.strip())
            if match is None:
                break
            dims = match[2].rstrip("x").split("x") if match[2] else []
            args.append({"position": len(args), "name": match[1], "shape": dims,
                         "dtype": match[3], "storage_dtype": match[3],
                         "line": masked.count("\n", 0, offset) + 1,
                         "dynamic_axes": [i for i, d in enumerate(dims) if d == "?"]})
            offset += len(part) + 1
        else:
            result["signature"] = args
    original_lines = source.splitlines()
    for number, line in enumerate(masked.splitlines(), 1):
        match = OP.match(line)
        if match:
            result["events"].append({"line": number, "operation": match[1],
                                     "text": original_lines[number - 1].strip()})
    result["operation_counts"] = dict(Counter(e["operation"] for e in result["events"]))
    return result


def _check(identifier, status, scope, detail, evidence=None):
    return {"id": identifier, "status": status, "scope": scope,
            "detail": detail, "evidence": evidence or []}


def _signature_check(left, right, facts, records):
    a, b = facts[left]["signature"], facts[right]["signature"]
    identifier = f"P0-{left[:2]}-{right[:2]}"
    locations = [{"stage": stage, "file": records[stage]["file"],
                  "lines": [arg["line"] for arg in facts[stage]["signature"] or []]}
                 for stage in (left, right)]
    if a is None or b is None:
        return _check(identifier, "uncertain", "入口存储形状/位宽的有限格式观察",
                      "至少一侧参数未完整识别；需要正式解析或明确参数映射。", locations)
    def signature(args):
        return [(tuple("?" if i in p["dynamic_axes"] else d for i, d in enumerate(p["shape"])),
                 p["storage_dtype"]) for p in args]
    if signature(a) != signature(b):
        return _check(identifier, "uncertain", "按参数位置比较入口表示",
                      "观察到参数数量、形状或元素位宽差异；缺少转换契约，不能直接判为语义错误。", locations)
    return _check(identifier, "match", "按参数位置比较 rank、静态维度、元素存储类型",
                  "有限格式下这些入口表示一致；未证明参数语义映射、动态维度关系、stride、别名或整数符号解释。", locations)


def _verify_mlir(group, executable):
    path = shutil.which(str(executable))
    if path is None:
        return {"status": "unsupported", "reason": f"MLIR verifier 不可执行：{executable}", "runs": []}
    runs = []
    for stage in STAGES[3:]:
        command = [path, "--verify-each", "-o", "/dev/null"]
        try:
            proc = subprocess.run(command, input=group.text[stage], text=True, capture_output=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as exc:
            runs.append({"stage": stage, "command": command, "error": f"{type(exc).__name__}: {exc}"})
            return {"status": "error", "runs": runs, "reason": f"MLIR 验证命令未完成：{exc}"}
        runs.append({"stage": stage, "command": command, "returncode": proc.returncode,
                     "stdout": proc.stdout, "stderr": proc.stderr})
    if any(run["returncode"] != 0 for run in runs):
        return {"status": "error", "runs": runs,
                "reason": "MLIR 验证命令未成功；需区分 IR 问题、方言/工具链不匹配与环境错误，不自动归为语义 mismatch。"}
    return {"status": "passed", "runs": runs,
            "scope": "仅验证 MLIR 格式及注册操作约束，不证明与 TIR 语义等价。"}


def _finish(result):
    result["summary"] = dict(Counter(c["status"] for c in result["checks"]))
    for status in ("match", "mismatch", "uncertain"):
        result["summary"].setdefault(status, 0)
    result["exit_code"] = EXIT_CODES[result["tool_status"]]
    result['findings_summary'] = dict(Counter(f['status'] for f in result['findings']))
    return result


def analyze(trace_dir, *, dynamic=False, semantic=False, case=None, mlir_opt=None,
            seed=0, timeout=120, max_steps=2_000_000, stages=None, translation=None,
            tir_backend='ast', tir_python=None,mlir_backend='text',ir_python=None):
    result = {
        "schema_version": 2, "tool_version": __version__,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "tool_status": "ok", "semantic_conclusion": "not_established",
        "input": {"root": str(Path(trace_dir).expanduser().resolve())},
        "checks": [], "facts": {}, "transitions": [], "diagnostics": [],
        "findings": [], "rule_facts": {}, "support_inventory": {},
        "capabilities": {
            "trace_integrity": "implemented", "tir_ast_observer": "implemented",
            "mlir_text_observer": "implemented", "formal_tir_verifier": "not_requested",
            "formal_mlir_verifier": "not_requested", "cross_stage_dataflow": "available_bounded_subset_not_requested",
            "official_mlir_object_frontend":"not_requested",
            "independent_dynamic_execution": "not_requested",
            "bounded_interpreters": "available_with_numpy_and_case",
        },
        "dynamic": {"status": "not_requested", "executed": False},
        "historical_execution": None,
        "limits": [
            "本报告是文件检查与局部事实观察，不是数学语义等价证明。",
            "Python AST 不等于 TVM TIR verifier；MLIR 文本索引不等于正式 parser。",
            "受限解释器只覆盖明确支持的 IR 操作；未实现任意 IR 或未知调用；view/一维物理线程/同步仅覆盖明确列出的子集。",
            "符号检查逐元素枚举当前有限形状；表达式相同仅在解释模型和契约内成立，表达式不同仍可能数学等价。",
            "原有测试结果只作为历史元数据，不计入本次动态验证。",
        ],
    }
    try:
        if tir_backend not in {'ast','tvm'} or (tir_python is not None and tir_backend != 'tvm'):
            raise InputError('tir_backend 必须为 ast/tvm；tir_python 仅用于 tvm 后端')
        if mlir_backend not in {'text','official'}:
            raise InputError('mlir_backend 必须为 text/official')
        if ir_python and tir_backend!='tvm' and mlir_backend!='official':
            raise InputError('ir_python 需要至少一个官方前端')
        if ir_python and tir_python and str(ir_python)!=str(tir_python):
            raise InputError('ir_python 与 tir_python 不能配置为不同解释器')
        if case is not None and not (dynamic or semantic):
            raise InputError("--case 必须配合 --dynamic 或 --semantic")
        if translation is not None and not (dynamic and case is not None):
            raise InputError("--translation 必须配合 --dynamic 和 --case")
        group = load_trace(trace_dir)
        result["input"] = group.public()
        result["historical_execution"] = {
            "source": group.metadata_source, "executed_in_this_check": False,
            "record": group.metadata.get("execution", group.metadata.get("runtime_validation")),
            "comparisons": group.metadata.get("comparisons"),
        }
        records = {stage["id"]: stage for stage in group.stages}
        verified = all(set(stage["verified_fields"]) == {"bytes", "lines", "sha256"}
                       for stage in group.stages)
        result["checks"].append(_check("I0-files", "match", "五阶段文件可读且非空", "已读取五份 UTF-8 阶段文本。"))
        result["checks"].append(_check("I1-integrity", "match" if verified else "uncertain",
            "对照已有清单的文件完整性", "五阶段字节数、行数与 SHA-256 均符合清单；清单本身不是独立来源认证。"
            if verified else "缺少清单或部分校验字段，只记录实际指纹，不能宣称完整性已核验。"))
        for index, stage in enumerate(STAGES):
            facts = tir_facts(group.text[stage]) if index < 3 else mlir_facts(group.text[stage])
            result["facts"][stage] = {"file": records[stage]["file"], **facts}
        rules = analyze_rules(group, result['facts'], mlir_backend=mlir_backend,tir_backend=tir_backend)
        result['rule_facts'] = rules['facts']
        result['support_inventory'] = rules['support']
        result['findings'].extend(rules['findings'])
        for left, right in zip(STAGES, STAGES[1:]):
            result["transitions"].append({"from": left, "to": right,
                "byte_identical": records[left]["sha256"] == records[right]["sha256"],
                "note": "文本是否相同；文本变化不自动判为语义变化。"})
            result["checks"].append(_signature_check(left, right, result["facts"], records))
            contract = result['checks'][-1]
            if contract['status'] != 'match':
                result['findings'].append(finding('CONTRACT-SIGNATURE', 'uncertain', 'static', contract['detail'],
                    locations=[{'stage': s, 'line': (result['facts'][s]['functions'] or [{}])[0].get('line')}
                               for s in (left, right)], details={'left': result['facts'][left]['signature'],
                                                                'right': result['facts'][right]['signature']}))
        for identifier, scope, detail in [
            ("P1-formula", "数学公式、初始化与下标方向", "需要 Agent 追踪完整计算；没有调用 mHC 专用规则。"),
            ("P2-dataflow", "全局输入到临时区再到输出", "局部事实不构成来源与写回覆盖证明。"),
            ("P3-precision", "类型转换的位置与舍入顺序", "已提供调用/操作行号；没有以转换指令数量推断精度正确。"),
            ("P4-control", "守卫、并行依赖与同步", "逻辑循环与物理线程映射仍需独立证据。"),
        ]:
            result["checks"].append(_check(identifier, "uncertain", scope, detail))
        if mlir_opt is not None:
            verification = _verify_mlir(group, mlir_opt)
            result["mlir_verification"] = verification
            result["capabilities"]["formal_mlir_verifier"] = verification["status"]
            if verification["status"] == "passed":
                for stage in STAGES[3:]:
                    result["facts"][stage]["formal_ir_verified"] = True
            if verification["status"] != "passed":
                result["tool_status"] = "unsupported" if verification["status"] == "unsupported" else "tool_error"
                result["diagnostics"].append(verification["reason"])
                result['findings'].append(finding('VERIFY-MLIR',
                    'unsupported' if verification['status']=='unsupported' else 'tool_error',
                    'formal_verifier', verification['reason'], details={'runs':verification['runs']}))
        if dynamic or semantic or tir_backend == 'tvm' or mlir_backend=='official':
            execution=request_dynamic(group,case,dynamic=dynamic,seed=seed,stages=stages,
                                      timeout=timeout,max_steps=max_steps,tir_backend=tir_backend,tir_python=tir_python,
                                      mlir_backend=mlir_backend,ir_python=ir_python)
            result["_evidence"]=execution.pop("_evidence",{})
            verification = execution.get('tir_verification')
            if verification is not None:
                verification['logs'] = {key:execution[key] for key in
                    ('worker_log','worker_stdout','worker_stderr') if execution.get(key)}
                result['tir_verification'] = verification
                result['capabilities']['formal_tir_verifier'] = verification['status']
                for run in verification['runs']:
                    stage = run['stage']
                    result['facts'][stage]['formal_ir_verified'] = run['status'] == 'passed'
                    if run['status'] != 'passed':
                        result['findings'].append(finding('VERIFY-TIR',
                            'tool_error' if run['status']=='invalid_ir' else 'unsupported',
                            'formal_verifier', run.get('diagnostic','官方 TIR 检查未通过；见 checks 原始诊断'),
                            stage=stage,details=run,
                            scope='官方结构检查结果；不自动归因为 lowering 语义错误。'))
                if verification['status'] != 'passed' and result['tool_status'] == 'ok':
                    result['tool_status'] = 'tool_error' if verification['status'] == 'error' else 'unsupported'
            elif tir_backend == 'tvm':
                result['capabilities']['formal_tir_verifier'] = 'not_completed'
            mlir_verification=execution.get('mlir_python_verification')
            if mlir_verification is not None:
                mlir_verification['logs']={k:execution[k] for k in ('worker_log','worker_stdout','worker_stderr') if execution.get(k)}
                result['mlir_python_verification']=mlir_verification
                result['capabilities']['official_mlir_object_frontend']=mlir_verification['status']
                if mlir_opt is None: result['capabilities']['formal_mlir_verifier']=mlir_verification['status']
                for run in mlir_verification['runs']:
                    result['facts'][run['stage']]['official_object_verified']=run['status']=='passed'
                    if run['status']=='passed': result['facts'][run['stage']]['formal_ir_verified']=True
                    else:
                        result['findings'].append(finding('VERIFY-MLIR-PYTHON',
                            'tool_error' if run['status']=='invalid_ir' else 'unsupported','formal_verifier',
                            run.get('diagnostic','官方 MLIR 验证未通过；见原始诊断'),stage=run['stage'],details=run))
                if mlir_verification['status']!='passed' and result['tool_status']=='ok':
                    result['tool_status']='tool_error' if mlir_verification['status']=='error' else 'unsupported'
            elif mlir_backend=='official':
                result['capabilities']['official_mlir_object_frontend']='not_completed'
            if dynamic or semantic:
                result["dynamic"]=execution
                merge_execution(result,group,execution,dynamic)
            elif execution['status'] != 'passed':
                result['tool_status'] = 'unsupported' if execution['status']=='unsupported' else 'tool_error'
                if execution.get('reason'):
                    result['diagnostics'].append(execution['reason'])
                    result['findings'].append(finding('VERIFY-FRONTEND-ENVIRONMENT',result['tool_status'],
                        'formal_verifier',execution['reason']))
        if translation is not None:
            supplement = request_dynamic(group, case, dynamic=True, seed=seed, stages=stages,
                                         timeout=timeout, max_steps=max_steps, translation=translation)
            original_binding = result['dynamic'].get('cases_fingerprint')
            translated_binding = supplement.get('cases_fingerprint')
            supplement['input_binding'] = ('identical' if original_binding and original_binding == translated_binding
                                            else 'original_inputs_unavailable')
            if original_binding and translated_binding and original_binding != translated_binding:
                supplement['input_binding'] = 'different'
                supplement['status'] = 'error'
                supplement['semantic_conclusion'] = 'input_binding_failed'
                supplement['reason'] = '两轮适配产生的输入、参考或比较契约不一致，不能视为同输入补充验证'
            result['agent_translation'] = supplement
            # Translation evidence must never overwrite the interpreter evidence or verdict.
            result.setdefault('_evidence', {}).update({
                'agent_translation/'+name: data
                for name, data in supplement.pop('_evidence', {}).items()})
            result['findings'].extend(supplement.get('findings', []))
            if supplement.get('reason'):
                result['findings'].append(finding('AGENT-TRANSLATION-EXECUTION', 'uncertain',
                    'agent_translation', supplement['reason']))
        locate(result['findings'], group)
    except InputError as exc:
        result["tool_status"] = "input_error"
        result["diagnostics"].append(str(exc))
        result['findings'].append(finding('INPUT-TRACE', 'input_error', 'input_validation', str(exc)))
    except (OSError, ValueError, TypeError, RecursionError, subprocess.SubprocessError) as exc:
        result["tool_status"] = "tool_error"
        result["diagnostics"].append(f"{type(exc).__name__}: {exc}")
        result['findings'].append(finding('TOOL-ANALYSIS', 'tool_error', 'tool', f'{type(exc).__name__}: {exc}'))
    return _finish(result)


def merge_execution(result,group,execution,requested_dynamic):
    """Preserve static observations, add scoped symbolic and numeric evidence."""
    status=execution['status']
    result['findings'].extend(execution.get('findings', []))
    code_status={'passed':'ok','mismatch':'mismatch','unsupported':'unsupported','error':'tool_error'}[status]
    if result['tool_status']=='ok':
        result['tool_status']=code_status
    if execution.get('reason'):
        result['diagnostics'].append(execution['reason'])
        result['findings'].append(finding('EXEC-REQUEST', 'tool_error' if status == 'error' else 'unsupported',
            'execution', execution['reason']))
    records={s['id']:s for s in group.stages}
    for number,item in enumerate(execution.get('symbolic',[])):
        scope=('有限形状逐元素公式、来源与精度表达式' if item.get('kind')=='expression'
               else '当前形状输出覆盖/解释模型')
        evidence=[]
        if item.get('kind')=='expression':
            difference=item.get('first_difference') or {}
            for side,key in [('left','from'),('right','to')]:
                stage=item[key]
                line=difference.get(side+'_writer_line')
                evidence.append({'stage':stage,'file':records[stage]['file'],'lines':[line] if line else []})
        elif item.get('stage'):
            stage=item['stage']
            evidence=[{'stage':stage,'file':records[stage]['file'],'lines':item.get('writer_lines',[])}]
        check_status=item['status'] if item['status'] in {'match','mismatch','uncertain'} else 'uncertain'
        detail=item.get('reason',f"用例 {item.get('case')}；详见 dynamic.symbolic[{number}]；有限枚举结果不推广到其他形状。")
        result['checks'].append(_check(f'S-{number:03}',check_status,scope,detail,evidence))
    for stage in STAGES:
        tests=[t for t in execution.get('tests',[]) if t['stage']==stage]
        if not tests:
            continue
        check_status=('mismatch' if any(t['status']=='mismatch' for t in tests) else
                      ('uncertain' if any(t['status']=='unsupported' for t in tests) else 'match'))
        result['checks'].append(_check('D-'+stage[:2],check_status,'当前测试集数值输出 vs 独立参考',
            f"{len(tests)} 次输出比较；解释/原生后端和逐项反例见 dynamic.tests。",
            [{'stage':stage,'file':records[stage]['file'],'lines':[]}]))
    if execution.get('symbolic'):
        result['capabilities']['cross_stage_dataflow']='bounded_element_provenance'
    result['capabilities']['independent_dynamic_execution']=status if requested_dynamic else 'not_requested'
    if execution.get('has_mismatch'):
        result['semantic_conclusion']='bounded_counterexample'
    elif status=='passed':
        unresolved=any(i['status']!='match' for i in execution.get('symbolic',[]))
        result['semantic_conclusion']=(('samples_passed_symbolic_unresolved' if requested_dynamic else 'bounded_expressions_unresolved') if unresolved else
                                       ('bounded_checks_passed' if requested_dynamic else 'bounded_expressions_match'))

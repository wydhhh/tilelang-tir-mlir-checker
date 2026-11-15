"""Command interface: one JSON response on stdout; diagnostics in the result."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path

from . import __version__, check
from .trace_io import InputError, excerpt


def main(argv=None):
    parser = argparse.ArgumentParser(description="TileLang TIR→MLIR：完整性、事实索引、受限 IR 语义与动态对照。")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="从单算子适配文件的真实编译中导出五阶段")
    export.add_argument("--adapter", required=True)
    export.add_argument("--out", required=True)
    export.add_argument("--timeout", type=int, default=600)
    pipeline = commands.add_parser("run", help="单算子：真实导出 → Checker → 返回 Agent 证据")
    pipeline.add_argument("--adapter", required=True, help="make_kernel() + make_cases(seed)")
    pipeline.add_argument("--operator", help="算子展示名称，用于报告和证据文件名")
    pipeline.add_argument("--out", required=True, help="新的运行目录")
    pipeline.add_argument("--static-only", action="store_true")
    pipeline.add_argument("--seed", type=int, default=0)
    pipeline.add_argument("--timeout", type=int, default=600)
    pipeline.add_argument("--max-steps", type=int, default=2_000_000)
    pipeline.add_argument("--mlir-opt")
    pipeline.add_argument("--tir-backend",choices=['ast','tvm'],default='ast',help="tvm: 官方读取/检查 + 受限对象节点模拟")
    pipeline.add_argument("--tir-python",help="有匹配 TileLang/TVM 的本机 Python；不连接服务器")
    pipeline.add_argument('--mlir-backend',choices=['text','official'],default='text',help='official: 官方 MLIR 对象解析/验证与受限模拟')
    pipeline.add_argument('--ir-python',help='同时具备所选官方前端的本机 Python；与tir-python兼容保留')
    run = commands.add_parser("check", help="检查一个算子的五阶段 Trace")
    run.add_argument("--trace", required=True, help="一个算子组目录，不是整个批次")
    run.add_argument("--operator", help="算子展示名称；省略时从manifest或IR识别")
    run.add_argument("--out", help="新的报告目录；默认在系统临时目录生成")
    run.add_argument("--dynamic", action="store_true", help="执行五阶段与参考实现对照，同时做有限形状符号检查")
    run.add_argument("--semantic", action="store_true", help="只做有限形状的逐元素表达式/覆盖检查，需要 --case")
    run.add_argument("--case", help="显式测试适配文件：make_cases(seed)；在隔离子进程执行")
    run.add_argument("--translation", help="Agent生成的阶段Python翻译；配合--dynamic/--case，补充结果单独记录")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--timeout", type=int, default=120, help="执行子进程的总超时秒数，1–3600")
    run.add_argument("--max-steps", type=int, default=2_000_000, help="每次单阶段解释的步数上限")
    run.add_argument("--mlir-opt", help="显式指定兼容的 mlir-opt，验证 03/04；无转换 passes")
    run.add_argument("--tir-backend",choices=['ast','tvm'],default='ast',help="tvm: 官方读取/检查 + 受限对象节点模拟；不回退 AST")
    run.add_argument("--tir-python",help="有匹配 TileLang/TVM 的本机 Python；由Agent配置环境")
    run.add_argument('--mlir-backend',choices=['text','official'],default='text',help='official: 不回退文本解析')
    run.add_argument('--ir-python',help='同时具备所选官方前端的本机 Python；由Agent配置环境')
    show = commands.add_parser("show", help="校验原始文件后，按行号返回一个阶段的片段")
    show.add_argument("--trace", required=True)
    show.add_argument("--stage", required=True, choices=["00", "01", "02", "03", "04"])
    show.add_argument("--start", type=int, default=1)
    show.add_argument("--end", type=int, default=80)
    args = parser.parse_args(argv)
    try:
        if args.command in {"export", "run"}:
            from .capture import export_trace
            out = Path(args.out).expanduser().resolve()
            if args.command == 'run' and args.operator is not None:
                from .naming import validate_operator_name
                validate_operator_name(args.operator)
            if out.exists():
                raise InputError("运行目录必须不存在")
            trace = out if args.command == "export" else out / "trace"
            response = export_trace(args.adapter, trace, timeout=args.timeout)
            code = response["exit_code"]
            if args.command == "run" and code == 0:
                captured = response
                result = check(trace, out_dir=out, dynamic=not args.static_only,
                               case=None if args.static_only else args.adapter,
                               seed=args.seed, timeout=args.timeout, max_steps=args.max_steps,
                               mlir_opt=args.mlir_opt, operator_name=args.operator,
                               tir_backend=args.tir_backend,tir_python=args.tir_python,
                               mlir_backend=args.mlir_backend,ir_python=args.ir_python)
                response = _response(result)
                response["capture"] = captured
                code = result["exit_code"]
        elif args.command == "show":
            response = excerpt(args.trace, args.stage, args.start, args.end)
            code = 0
        else:
            # Reserve a unique parent; report writer itself requires a new folder.
            out = args.out or str(Path(tempfile.mkdtemp(prefix="tir-mlir-check-")) / "result")
            result = check(args.trace, out_dir=out, dynamic=args.dynamic,semantic=args.semantic,
                           case=args.case, mlir_opt=args.mlir_opt,
                           seed=args.seed,timeout=args.timeout,max_steps=args.max_steps,operator_name=args.operator,
                           translation=args.translation,tir_backend=args.tir_backend,tir_python=args.tir_python,
                           mlir_backend=args.mlir_backend,ir_python=args.ir_python)
            response = _response(result)
            code = result["exit_code"]
    except InputError as exc:
        code = 2
        response = {"tool_status": "input_error", "exit_code": code, "diagnostics": [str(exc)]}
    except Exception as exc:
        code = 3
        traceback.print_exc(file=sys.stderr)
        response = {"tool_status": "tool_error", "exit_code": code,
                    "diagnostics": [f"{type(exc).__name__}: {exc}"]}
    print(json.dumps(response, ensure_ascii=False))
    return code


def _response(result):
    response = {key: result[key] for key in [
        "schema_version", "tool_version", "tool_status", "exit_code",
        "semantic_conclusion", "summary", "diagnostics", "artifacts",
        "findings_summary", "support_inventory"]}
    response["dynamic_status"] = result["dynamic"]["status"]
    response['tir_verification'] = result.get('tir_verification',{'status':result['capabilities']['formal_tir_verifier']})
    response['mlir_python_verification']=result.get('mlir_python_verification',{'status':result['capabilities']['official_mlir_object_frontend']})
    response["agent_translation_status"] = result.get("agent_translation", {}).get("status", "not_requested")
    if 'agent_translation' in result:
        supplement = result['agent_translation']
        response['agent_translation'] = {key: supplement.get(key) for key in
            ['status', 'input_binding', 'fidelity', 'semantic_conclusion', 'reason']}
        response['agent_translation']['comparison_count'] = len(supplement.get('tests', []))
        response['agent_translation']['matched_comparisons'] = sum(
            t['status'] == 'match' for t in supplement.get('tests', []))
    response["operator"] = result.get("operator")
    response["ir"] = result["input"].get("stages", [])
    response["agent_diagnostics"] = result.get("agent_diagnostics", [])[:30]
    response["diagnostic_count"] = len(result.get("agent_diagnostics", []))
    response["next_step"] = "读取自动记录并用show回查IR；Agent报告包含检查结论、测试范围与结果、问题与Agent判断三部分。证据位置以artifacts.evidence为准。"
    return response


if __name__ == "__main__":
    raise SystemExit(main())

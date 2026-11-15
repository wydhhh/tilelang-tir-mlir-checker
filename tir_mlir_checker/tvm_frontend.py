"""Optional, version-matched TVM loading/verifiers. Imported only in a worker.

TVMScript parsing uses TVM's Python frontend, not a security sandbox. Only use
trusted captures. No compiler, target or source rewriting is performed here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .trace_io import STAGES


def identity(tvm):
    return {"version": tvm.__version__,
            "git_commit": tvm.support.libinfo().get("GIT_COMMIT_HASH"),
            "python_module": str(Path(tvm.__file__).resolve())}


def load_stages(group):
    """Return actual PrimFuncs and per-stage diagnostics, never AST fallbacks."""
    result = {"status": "passed", "runs": [], "scope":
              "官方 SSA/结构检查；不证明边界安全、数值等价或真机正确。"}
    functions = {}
    try:
        from tilelang import tvm
        result["environment"] = identity(tvm)
    except (ImportError, OSError) as exc:
        result.update(status="unsupported", reason=f"匹配的 TileLang/TVM 不可用：{exc}")
        result["runs"] = [{"stage": s, "status": "unsupported", "failure_kind": "dependency_unavailable",
                           "diagnostic": str(exc)} for s in STAGES[:3]]
        return functions, result
    for stage in STAGES[:3]:
        run = {"stage": stage, "status": "unsupported", "checks": [], "text_line_mapping": "unavailable"}
        result["runs"].append(run)
        entry = group.metadata.get("tir_objects", {}).get(stage)
        try:
            if entry:
                for field, actual in (("version", result["environment"]["version"]),
                                      ("git_commit", result["environment"]["git_commit"])):
                    expected = entry.get("tvm", {}).get(field)
                    if expected and expected != actual:
                        run.update(failure_kind="version_incompatible", diagnostic=f"TVM {field}: expected {expected}, actual {actual}")
                        break
                else:
                    obj = tvm.ir.load_json((group.root / entry["file"]).read_text())
                    run.update(loader="tvm.ir.load_json", object_sha256=entry["sha256"])
                    # The object and text must describe the same actual capture.
                    canonical = obj.script()
                    if hashlib.sha256(canonical.encode()).hexdigest() != entry["text_sha256"]:
                        run.update(failure_kind="object_text_incompatible", diagnostic="对象打印与绑定阶段不一致；未静默改用文本解释。")
                    else:
                        run["binding"] = "object_canonical_text_sha256"
                        run["status"] = "loaded"
                if run["status"] != "loaded":
                    continue
            else:
                # The official parser may not support all fork-specific extensions.
                from tvm.script import ir as I, tir as T
                obj = tvm.script.from_source(group.text[stage], extra_vars={"I": I, "T": T})
                run.update(status="loaded", loader="tvm.script.from_source", binding="stage_text")
            if isinstance(obj, tvm.tir.PrimFunc):
                fun = obj
            elif isinstance(obj, tvm.IRModule) and len(obj.functions) == 1:
                fun = list(obj.functions.values())[0]
            else:
                run.update(status="unsupported", failure_kind="function_contract", diagnostic="需要单个 TIR PrimFunc。")
                continue
            if not isinstance(fun, tvm.tir.PrimFunc):
                run.update(status="unsupported", failure_kind="function_contract", diagnostic="IR 不是 TIR PrimFunc。")
                continue
            for name in ("verify_ssa", "verify_well_formed"):
                checker = getattr(tvm.tir.analysis, name, None)
                if checker is None:
                    run["checks"].append({"name": name, "status": "unsupported", "diagnostic": "当前 TVM 未提供接口"})
                    continue
                try:
                    ok = bool(checker(fun))
                    run["checks"].append({"name": name, "status": "passed" if ok else "invalid_ir",
                                          "returned": ok})
                except Exception as exc:
                    run["checks"].append({"name": name, "status": "invalid_ir",
                                          "diagnostic": f"{type(exc).__name__}: {exc}"})
            states = {c["status"] for c in run["checks"]}
            run["status"] = "invalid_ir" if "invalid_ir" in states else ("unsupported" if "unsupported" in states else "passed")
            if run["status"] == "passed":
                functions[stage] = fun
        except Exception as exc:
            run.update(status="unsupported", failure_kind="object_or_parser_incompatible",
                       diagnostic=f"{type(exc).__name__}: {exc}")
    states = {r["status"] for r in result["runs"]}
    result["status"] = "error" if "invalid_ir" in states else ("unsupported" if states != {"passed"} else "passed")
    return functions, result


def verify_only(request):
    from .trace_io import load_trace
    _, verification = load_stages(load_trace(request["trace"]))
    return {"status": verification["status"], "executed": False, "tir_verification": verification}

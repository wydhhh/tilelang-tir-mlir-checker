"""Hooks for the tested TileLang RISC-V branch, never installed into TileLang."""
from __future__ import annotations

import ast
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import runpy
import sys
import traceback
import uuid

from .trace_io import STAGES


class CaptureUnsupported(Exception):
    pass


def collect(request):
    from . import __version__
    if platform.machine() != "riscv64":
        raise CaptureUnsupported("当前采集适配器要求已有 TileLang RISC-V 环境；不支持其他后端。")
    if os.environ.get("LD_PRELOAD") or "libbf16shim" in Path("/proc/self/maps").read_text():
        raise CaptureUnsupported("采集前须排除 LD_PRELOAD/旧 bf16 shim，并显式配置所需工具链。")
    import tilelang
    from tilelang import tvm
    lower = importlib.import_module("tilelang.engine.lower")
    libgen = importlib.import_module("tilelang.jit.adapter.riscv.libgen")
    required = ((lower, "LowerAndLegalize"), (lower, "OptimizeForTarget"),
                (lower, "device_codegen"), (libgen, "_build_pipeline"),
                (libgen, "_PRE_LOW_PRECISION_LEGALIZATION_PASSES"))
    if any(not hasattr(m, a) for m, a in required):
        raise CaptureUnsupported("TileLang 编译接口与采集器不兼容；未猜测或重放五阶段。")
    out, work = Path(request["out"]), Path(request["work"])
    adapter = Path(request["adapter"])
    sys.path.insert(0, str(adapter.parent))
    namespace = runpy.run_path(str(adapter), run_name="checker_export_adapter")
    factory = namespace.get("make_kernel")
    options = namespace.get("compile_options", {})
    if not callable(factory) or not isinstance(options, dict):
        raise ValueError("适配文件须提供 make_kernel() 和可选 compile_options 字典")
    options = dict(options)
    if str(options.get("target", "riscv")) != "riscv":
        raise CaptureUnsupported("本版本只采集 target=riscv")
    options["target"] = "riscv"
    kernel = factory()
    if not isinstance(kernel, tvm.tir.PrimFunc):
        raise ValueError("make_kernel() 必须返回单个未编译的 TIR PrimFunc")
    records, metadata, seen, objects = [], {}, set(), {}
    from .tvm_frontend import identity
    tvm_identity = identity(tvm)

    def write(index, text, origin):
        if index in seen:
            raise CaptureUnsupported("一次适配只能编译一个算子，检测到重复阶段")
        seen.add(index)
        raw = str(text).encode("utf-8")
        filename = STAGES[index] + (".tir" if index < 3 else ".mlir")
        (out / filename).write_bytes(raw)
        records.append(dict(id=STAGES[index], file=filename, sha256=hashlib.sha256(raw).hexdigest(),
                            bytes=len(raw), lines=len(raw.splitlines()), origin=origin))

    def tir(index, mod, origin):
        write(index, mod.script(), origin)
        full = mod.script(show_meta=True)
        (work / (STAGES[index] + "_full.tir")).write_text(full)
        filename = STAGES[index] + "_object.json"
        raw = tvm.ir.save_json(mod).encode()
        (out / filename).write_bytes(raw)
        objects[STAGES[index]] = dict(file=filename, sha256=hashlib.sha256(raw).hexdigest(),
            text_sha256=records[-1]['sha256'], tvm=tvm_identity)
        for stmt in ast.parse(full).body:
            if (isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name)
                    and stmt.targets[0].id == "metadata" and isinstance(stmt.value, ast.Call)):
                metadata[STAGES[index]] = ast.literal_eval(stmt.value.args[0])

    legalize, optimize = lower.LowerAndLegalize, lower.OptimizeForTarget
    def legal(mod, target):
        tir(0, mod, "actual before LowerAndLegalize")
        result = legalize(mod, target)
        tir(1, result, "actual after LowerAndLegalize")
        return result
    def opt(mod, target):
        result = optimize(mod, target)
        tir(2, result, "actual after OptimizeForTarget")
        return result
    def wrap_codegen(original):
        def codegen(mod, target):
            result = original(mod, target)
            write(3, result.inspect_source(), "actual RISC-V codegen")
            return result
        return codegen
    lower.LowerAndLegalize, lower.OptimizeForTarget = legal, opt
    lower.device_codegen = wrap_codegen(lower.device_codegen)
    if hasattr(lower, "device_codegen_without_compile"):
        lower.device_codegen_without_compile = wrap_codegen(lower.device_codegen_without_compile)
    pipeline = libgen._build_pipeline
    passes = tuple(libgen._PRE_LOW_PRECISION_LEGALIZATION_PASSES)
    class Tap:
        def __init__(self, inner):
            self.inner = inner
        def run(self, source):
            captured = out / (STAGES[3] + ".mlir")
            if not captured.exists() or source != captured.read_text():
                raise CaptureUnsupported("pre-LLVM 输入与实际 Stage03 不一致，拒绝混用")
            result = self.inner.run(source)
            write(4, result, "actual pre-LLVM pipeline")
            return result
    def build(pass_list):
        result = pipeline(pass_list)
        return Tap(result) if tuple(pass_list) == passes else result
    libgen._build_pipeline = build
    os.environ["TILELANG_DISABLE_CACHE"] = "1"
    compiled = tilelang.compile(kernel, **options)
    close = getattr(compiled, "close", None)
    if close:
        close()
    if seen != set(range(5)):
        raise CaptureUnsupported(f"真实编译未产生五阶段，实际阶段：{sorted(seen)}")
    source_files = {}
    for module in (tilelang, lower, libgen):
        path = Path(module.__file__).resolve()
        source_files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    mlir_environment={}
    try:
        from .mlir_frontend import identity as mlir_identity
        mlir_environment=mlir_identity()
    except (ImportError,OSError):
        pass  # Capture remains usable without optional MLIR Python bindings.
    manifest = dict(schema_version=1, name=kernel.attrs.get("global_symbol", "kernel"),
                    run_id=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8],
                    tool_version=__version__, stages=sorted(records, key=lambda r: r["id"]),
                    tir_metadata=metadata, tir_objects=objects, mlir_environment=mlir_environment,stage04_pipeline=list(passes),
                    adapter=dict(file=str(adapter), sha256=hashlib.sha256(adapter.read_bytes()).hexdigest()),
                    compiler_sources=source_files,
                    environment={k: os.getenv(k) for k in ("TILELANG_RISCV_LLVM_ROOT", "TILELANG_DISABLE_CACHE",
                                                           "TILELANG_RISCV_ABI", "LD_PRELOAD")})
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return {"tool_status": "ok", "exit_code": 0, "run_id": manifest["run_id"], "diagnostics": []}


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    try:
        result = collect(request)
    except (CaptureUnsupported, ImportError) as exc:
        result = {"tool_status": "unsupported", "exit_code": 4, "diagnostics": [str(exc)]}
        traceback.print_exc()
    except Exception as exc:
        result = {"tool_status": "tool_error", "exit_code": 3,
                  "diagnostics": [f"{type(exc).__name__}: {exc}"]}
        traceback.print_exc()
    Path(request["response"]).write_text(json.dumps(result, ensure_ascii=False))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())

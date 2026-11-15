"""Collect actual compilation checkpoints in a disposable process."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import zipfile

from .trace_io import InputError, load_trace


def export_trace(adapter, out_dir, *, timeout=600):
    adapter = Path(adapter).expanduser().resolve()
    out = Path(out_dir).expanduser().resolve()
    if not adapter.is_file():
        raise InputError(f"适配文件不存在：{adapter}")
    if out.exists():
        raise InputError("导出目录必须不存在，避免混用 Trace")
    if not 1 <= timeout <= 3600:
        raise InputError("timeout 必须为 1..3600")
    out.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="checker-capture-") as temp:
        work = Path(temp)
        request = work / "request.json"
        response = work / "response.json"
        request.write_text(json.dumps({"adapter": str(adapter), "out": str(out),
                                      "response": str(response), "work": str(work)}))
        env = dict(os.environ, TILELANG_DISABLE_CACHE="1", PYTHONDONTWRITEBYTECODE="1")
        root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        with (work / "compiler.log").open("w") as log:
            proc = subprocess.Popen([sys.executable, "-B", "-m", "tir_mlir_checker.capture_worker",
                                     str(request)], stdout=log, stderr=subprocess.STDOUT,
                                    env=env, start_new_session=True)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        result = (json.loads(response.read_text()) if response.exists() else
                  {"tool_status": "tool_error", "exit_code": 3,
                   "diagnostics": ["采集子进程超时或异常退出；保留编译日志和部分阶段，不能视为完整导出。"]})
        if result["exit_code"] == 0:
            group = load_trace(out)
            result["ir"] = group.stages
        result["artifacts"] = {"trace": str(out), "capture_evidence": str(out / "采集证据.zip")}
        result["agent_diagnostics"] = [
            {"id": f"EXPORT-{i + 1:03d}",
             "severity": "remark" if result["tool_status"] == "unsupported" else "error",
             "rule_id": "CAP-EXPORT" if result["tool_status"] == "unsupported" else "TOOL-EXPORT",
             "status": result["tool_status"], "message": message, "location": None,
             "notes": [], "evidence_ref": f"采集证据.zip!capture.json#/diagnostics/{i}"}
            for i, message in enumerate(result["diagnostics"])
        ]
        with zipfile.ZipFile(out / "采集证据.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(adapter, "adapter.py")
            for p in work.iterdir():
                if p.is_file() and p.name != "request.json":
                    archive.write(p, p.name)
            archive.writestr("capture.json", json.dumps(result, ensure_ascii=False, indent=2))
        return result

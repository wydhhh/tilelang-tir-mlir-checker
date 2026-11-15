"""Read existing exports without executing TVMScript or extracting ZIPs."""
from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

STAGES = (
    "00_tir_initial", "01_tir_legalized", "02_tir_optimized",
    "03_mlir_linalg", "04_mlir_final",
)


class InputError(ValueError):
    """Missing, ambiguous, malformed, or integrity-inconsistent input."""


@dataclass
class TraceGroup:
    root: Path
    metadata: dict
    metadata_source: str | None
    run_id: str | None
    stages: list[dict]
    text: dict[str, str]

    def public(self):
        return {
            "root": str(self.root), "name": self.metadata.get("name", self.root.name),
            "run_id": self.run_id, "metadata_source": self.metadata_source,
            "config": self.metadata.get("config"), "metadata": self.metadata,
            "stages": self.stages,
        }


def _read_manifest(path, zipped=False):
    try:
        if zipped:
            with zipfile.ZipFile(path) as archive:
                entries = [info for info in archive.infolist() if info.filename == "manifest.json"]
                if len(entries) != 1 or entries[0].file_size > 16 * 1024 * 1024:
                    raise InputError(f"ZIP 中 manifest.json 缺失、重复或过大：{path}")
                value = json.loads(archive.read(entries[0]).decode("utf-8"))
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(value, dict) or type(value.get("schema_version")) is not int
                or value.get("schema_version") != 1):
            raise InputError(f"仅支持 schema_version=1 的 manifest：{path}")
        return value
    except (OSError, UnicodeError, ValueError, zipfile.BadZipFile, RuntimeError) as exc:
        if isinstance(exc, InputError):
            raise
        raise InputError(f"无法读取清单 {path}：{exc}") from exc


def _metadata(root):
    direct = root / "manifest.json"
    if direct.exists():
        manifest = _read_manifest(direct)
        if "groups" in manifest:
            raise InputError("请指定一个算子组目录，而非整个批次目录")
        return manifest, root, str(direct), manifest.get("run_id")
    archive = root.parent / "验证与来源.zip"
    if archive.exists():
        manifest = _read_manifest(archive, zipped=True)
        groups = manifest.get("groups")
        if not isinstance(groups, list) or not all(isinstance(g, dict) for g in groups):
            raise InputError("批次 manifest.groups 格式错误")
        selected = [g for g in groups if g.get("name") == root.name]
        if len(selected) != 1:
            raise InputError(f"批次中无法唯一定位算子组：{root.name}")
        # Only metadata for this one group is exposed; historical execution is data.
        group = dict(selected[0])
        group["export_environment"] = manifest.get("environment")
        group["stage04_pipeline"] = manifest.get("stage04_pipeline")
        return group, root.parent, str(archive) + "!manifest.json", manifest.get("batch_id")
    return {}, root, None, None


def _stage_index(entry):
    if not isinstance(entry, dict):
        raise InputError("阶段清单条目必须为对象")
    if "id" in entry:
        if entry["id"] not in STAGES:
            raise InputError(f"未知阶段 id：{entry['id']}")
        return STAGES.index(entry["id"])
    index = entry.get("stage")
    if type(index) is not int or not 0 <= index < len(STAGES):
        raise InputError(f"未知阶段编号：{index}")
    return index


def load_trace(trace_dir):
    root = Path(trace_dir).expanduser().resolve()
    if not root.is_dir():
        raise InputError(f"Trace 目录不存在：{root}")
    metadata, base, source, run_id = _metadata(root)
    entries = metadata.get("stages")
    indexed = {}
    if source is not None:
        if not isinstance(entries, list):
            raise InputError("manifest 缺少 stages 列表")
        for entry in entries:
            index = _stage_index(entry)
            if index in indexed:
                raise InputError(f"阶段重复：{STAGES[index]}")
            indexed[index] = entry
        if len(indexed) != 5:
            raise InputError("manifest 必须包含且仅包含五个阶段")
    stages, text = [], {}
    for index, stage in enumerate(STAGES):
        meta = indexed.get(index, {})
        if source is None:
            candidates = [root / (stage + ext) for ext in
                          ((".tir", ".tvmscript") if index < 3 else (".mlir",))]
            existing = [p for p in candidates if p.is_file()]
            if len(existing) != 1:
                raise InputError(f"阶段缺失或同名文件有歧义：{stage}")
            path = existing[0].resolve()
        else:
            relative = meta.get("file")
            if not isinstance(relative, str) or Path(relative).is_absolute():
                raise InputError(f"阶段必须使用相对路径：{stage}")
            path = (base / relative).resolve()
        if not path.is_relative_to(root):
            raise InputError(f"阶段路径越出所选算子目录：{path}")
        try:
            raw = path.read_bytes()
            value = raw.decode("utf-8")
        except (OSError, UnicodeError) as exc:
            raise InputError(f"阶段不可读：{stage}：{exc}") from exc
        if not value.strip():
            raise InputError(f"阶段为空：{stage}")
        actual = {"sha256": hashlib.sha256(raw).hexdigest(),
                  "bytes": len(raw), "lines": len(value.splitlines())}
        verified = []
        for field, observed in actual.items():
            if field not in meta:
                continue
            expected = meta[field]
            if type(expected) is not type(observed) or expected != observed:
                raise InputError(f"{stage} 的 {field} 校验失败：记录 {expected}，实际 {observed}")
            verified.append(field)
        stages.append({"id": stage, "kind": "tir" if index < 3 else "mlir",
                       "file": str(path), **actual, "verified_fields": verified,
                       "origin": meta.get("origin", meta.get("producer"))})
        text[stage] = value
    objects = metadata.get('tir_objects', {})
    if not isinstance(metadata.get('mlir_environment',{}),dict):
        raise InputError('mlir_environment 必须为对象')
    if not isinstance(objects, dict) or any(s not in STAGES[:3] for s in objects):
        raise InputError('tir_objects 必须按 00/01/02 完整阶段 ID 索引')
    for stage, entry in objects.items():
        if not isinstance(entry, dict) or not isinstance(entry.get('file'), str):
            raise InputError(f'{stage}: TIR 对象记录格式无效')
        relative = Path(entry['file'])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root) or path == root:
            raise InputError(f'{stage}: TIR 对象路径越界')
        # Object sidecars may never overwrite a stage, the manifest or adapter.
        if relative.name != stage + '_object.json' or len(relative.parts) != 1:
            raise InputError(f'{stage}: 对象文件名须为 {stage}_object.json')
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise InputError(f'{stage}: TIR 对象不可读：{exc}') from exc
        if len(raw) > 32 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != entry.get('sha256'):
            raise InputError(f'{stage}: TIR 对象大小/sha256 校验失败')
        if entry.get('text_sha256') != stages[STAGES.index(stage)]['sha256']:
            raise InputError(f'{stage}: TIR 对象绑定的文本指纹不一致')
        if not isinstance(entry.get('tvm',{}),dict):
            raise InputError(f'{stage}: TIR 对象 TVM 身份无效')
    return TraceGroup(root, metadata, source, run_id, stages, text)


def excerpt(trace_dir, stage, start=1, end=80):
    """Return original, integrity-checked lines for Agent follow-up."""
    if not 1 <= start <= end or end - start >= 400:
        raise InputError("行号须为正且 start<=end，每次最多读取 400 行")
    if str(stage) not in {"00", "01", "02", "03", "04"}:
        raise InputError("stage 必须为 00、01、02、03 或 04")
    group = load_trace(trace_dir)
    key = STAGES[int(stage)]
    record = group.stages[int(stage)]
    lines = group.text[key].splitlines()
    if start > len(lines):
        raise InputError(f"起始行超过文件总行数 {len(lines)}")
    return {"tool_status": "ok", "stage": key, "file": record["file"],
            "sha256": record["sha256"], "total_lines": len(lines),
            "lines": [{"line": i, "text": lines[i - 1]}
                      for i in range(start, min(end, len(lines)) + 1)]}

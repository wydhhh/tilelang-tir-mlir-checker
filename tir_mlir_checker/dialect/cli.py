"""``tir-mlir-dialect``: dialect-level contract checking, separate from ``tir-mlir-check``."""
from __future__ import annotations

import argparse
import json
import sys
import traceback

from .. import __version__
from ..trace_io import InputError
from . import check_module, check_trace, inventory_module
from .model import STATUSES, DialectReadError, ProfileError
from .profile import template_from_module
from .reader import read_module

BACKENDS = ("auto", "official", "text")


def _dump(value):
    print(json.dumps(value, ensure_ascii=False))
    return value.get("exit_code", 0)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="tir-mlir-dialect",
        description="MLIR Dialect 层的契约检查：方言/操作/属性/类型/符号，不执行数值。")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    inventory = commands.add_parser("inventory", help="盘点模块中的方言、操作、符号与限制")
    inventory.add_argument("--module", required=True)
    inventory.add_argument("--backend", choices=BACKENDS, default="auto")
    inventory.add_argument("--no-verify", action="store_true")
    inventory.add_argument("--out")

    check = commands.add_parser("check", help="按契约 profile 检查模块或 Trace 阶段")
    source = check.add_mutually_exclusive_group(required=True)
    source.add_argument("--module")
    source.add_argument("--trace")
    check.add_argument("--stage", default="03", help="Trace 阶段：00–04 或完整 ID")
    contract = check.add_mutually_exclusive_group()
    contract.add_argument("--profile", default="generic", help="内置 profile 名")
    contract.add_argument("--profile-file", help="外部 JSON profile")
    check.add_argument("--backend", choices=BACKENDS, default="auto")
    check.add_argument("--no-verify", action="store_true")
    check.add_argument("--out")

    template = commands.add_parser("template", help="从实际模块生成契约骨架（全部 pending）")
    template.add_argument("--module", required=True)
    template.add_argument("--namespace", required=True)
    template.add_argument("--name")
    template.add_argument("--version", default="0.1")
    template.add_argument("--status", choices=STATUSES, default="pending")
    template.add_argument("--backend", choices=BACKENDS, default="auto")
    template.add_argument("--out")

    args = parser.parse_args(argv)
    try:
        if args.command == "inventory":
            result = inventory_module(args.module, backend=args.backend,
                                      verify=not args.no_verify, out=args.out)
            return _dump(result)
        if args.command == "check":
            kwargs = dict(profile=args.profile, profile_file=args.profile_file,
                          backend=args.backend, verify=not args.no_verify, out=args.out)
            if args.trace:
                result = check_trace(args.trace, stage=args.stage, **kwargs)
            else:
                result = check_module(args.module, **kwargs)
            return _dump(result)
        facts = read_module(args.module, backend=args.backend)
        raw = template_from_module(facts, namespace=args.namespace, name=args.name,
                                   version=args.version, status=args.status)
        from .profile import profile_from_dict
        profile_from_dict(raw)  # template must round-trip through validation
        if args.out:
            from pathlib import Path
            path = Path(args.out).expanduser()
            if path.exists():
                raise InputError(f"输出文件已存在：{path}")
            path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(raw, ensure_ascii=False, indent=2))
        return 0
    except (InputError, ProfileError, DialectReadError, FileExistsError) as exc:
        kind = getattr(exc, "kind", "input")
        code = 4 if kind == "dependency" else 2
        print(json.dumps({"tool_status": "input_error" if code == 2 else "unsupported",
                          "exit_code": code, "layer": "dialect",
                          "diagnostics": [str(exc)]}, ensure_ascii=False))
        return code
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"tool_status": "tool_error", "exit_code": 3, "layer": "dialect",
                          "diagnostics": [f"{type(exc).__name__}: {exc}"]},
                         ensure_ascii=False))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

"""Thin bridge to the official MLIR verifier.

The verifier answers "is this structurally well formed", not "is this
semantically what the target contract requires".  Diagnostics emitted during
verification are captured by the calling reader's diagnostic handler.
"""
from __future__ import annotations


def verify_module(module_op):
    """Return ``(bool|None, error_message|None)``.

    ``None`` means verification could not be completed (for example, the
    module contains unregistered operations); it is never reported as success.
    """
    try:
        return bool(module_op.verify()), None
    except Exception as exc:  # pragma: no cover - depends on bindings internals
        return None, f"{type(exc).__name__}: {exc}"

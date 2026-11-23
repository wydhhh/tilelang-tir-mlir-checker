"""Dialect-layer facts and contract data model.

The model is target-agnostic: a module is a flat list of dialect-qualified
operations plus symbols and limits; a profile supplies the target contract.
Nothing here parses MLIR or imports MLIR bindings.
"""
from __future__ import annotations

from dataclasses import dataclass, field

STATUSES = ("confirmed", "candidate", "pending")


class DialectReadError(ValueError):
    """Reading the module failed; ``kind`` selects the CLI exit class.

    kind: "input" (bad path/text), "parse" (MLIR rejected the text),
          "dependency" (requested backend is unavailable).
    """

    def __init__(self, message, *, kind="input"):
        super().__init__(message)
        self.kind = kind


class ProfileError(ValueError):
    """A contract profile is missing, malformed, or self-inconsistent."""


@dataclass(frozen=True)
class DialectOp:
    """One operation as observed by a reader."""

    name: str                       # canonical name, e.g. "linalg.matmul"
    dialect: str                    # namespace, e.g. "linalg"
    line: int | None = None
    file: str | None = None
    node_path: str | None = None    # reader-defined stable path in the module
    operands: tuple[str, ...] = ()  # printed type strings
    results: tuple[str, ...] = ()
    types: tuple[str, ...] = ()     # every type string observed on the operation
    attributes: dict[str, str] = field(default_factory=dict)
    region_count: int | None = None
    text: str = ""
    parent: str | None = None
    ir_location: str | None = None
    implicit: bool = False          # no printable source line (e.g. linalg.yield)
    source: str = "text"            # "text" | "official"

    def location(self, stage=None):
        loc = {"file": self.file, "line": self.line, "node_path": self.node_path,
               "node_frontend": "mlir"}
        if self.text:
            loc["excerpt"] = self.text
        if self.ir_location:
            loc["ir_location"] = self.ir_location
        if stage:
            loc["stage"] = stage
        return loc


@dataclass
class DialectModule:
    """A parsed module plus the evidence needed to qualify every conclusion."""

    path: str = ""
    sha256: str = ""
    backend: str = "text"           # "text" | "official"
    ops: list[DialectOp] = field(default_factory=list)
    symbol_defs: list[str] = field(default_factory=list)   # duplicates preserved
    callees: list[str] = field(default_factory=list)
    module_attributes: dict[str, str] = field(default_factory=dict)
    parse_mode: str = "strict"      # "strict" | "permissive"
    nonstandard_dialects: list[str] = field(default_factory=list)
    diagnostics: list[dict] = field(default_factory=list)   # verifier/parse messages
    limits: list[str] = field(default_factory=list)
    fallback_reason: str | None = None

    @property
    def entry_symbols(self):
        return sorted(set(self.symbol_defs))

    def dialects(self):
        return sorted({op.dialect for op in self.ops})

    def ops_by_dialect(self, namespace):
        return [op for op in self.ops if op.dialect == namespace]

    def public(self):
        return {
            "path": self.path, "sha256": self.sha256, "backend": self.backend,
            "parse_mode": self.parse_mode,
            "dialects": self.dialects(),
            "op_count": len(self.ops),
            "entry_symbols": self.entry_symbols,
            "nonstandard_dialects": self.nonstandard_dialects,
            "fallback_reason": self.fallback_reason,
            "limits": self.limits,
        }


@dataclass(frozen=True)
class OpContract:
    name: str
    status: str = "candidate"
    operands: int | None = None
    results: int | None = None
    regions: int | None = None
    attributes: tuple[str, ...] = ()
    allowed_attributes: tuple[str, ...] | None = None
    note: str = ""


@dataclass(frozen=True)
class DialectContract:
    namespace: str
    status: str = "candidate"
    closed_world: bool = False
    ops: dict[str, OpContract] = field(default_factory=dict)
    allowed_dtypes: tuple[str, ...] | None = None
    attributes_required: tuple[str, ...] = ()
    attributes_allowed: tuple[str, ...] | None = None
    note: str = ""


@dataclass(frozen=True)
class DtypePolicy:
    status: str = "pending"
    allowed: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class AlignmentPolicy:
    status: str = "pending"
    alloc_ops: tuple[str, ...] = ("memref.alloc", "memref.alloca")
    required_alignment_bytes: int | None = None
    note: str = ""


@dataclass(frozen=True)
class ContractProfile:
    name: str
    version: str
    status: str
    allowed_dialects: frozenset[str]
    forbidden_dialects: frozenset[str]
    required_dialects: frozenset[str]
    entry_symbol: str | None
    dialects: dict[str, DialectContract]
    dtype_policy: DtypePolicy | None
    alignment_policy: AlignmentPolicy | None
    stage_rules: dict[str, frozenset[str]]
    pending: tuple[str, ...]
    notes: tuple[str, ...]

    def public(self):
        return {"name": self.name, "version": self.version, "status": self.status,
                "pending_items": len(self.pending)}


def op_contract_status(op_contract):
    """Confirmed violations are errors; candidate/pending are uncertain."""
    return "mismatch" if op_contract.status == "confirmed" else "uncertain"

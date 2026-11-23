"""Built-in contract profiles and the standard-dialect baseline.

Profiles are data.  The AME profile is deliberately a *candidate*: the vendor
namespace, operation set, attributes, dtypes and alignment rules are not yet
confirmed, so every check on those items reports ``uncertain`` rather than
inventing a verdict.
"""
from __future__ import annotations

# Dialects this checker treats as ordinary MLIR dialects even without a profile
# clause.  Membership only suppresses "unknown dialect"; it does not grant
# permission (that is the profile's allowed/forbidden lists).
STANDARD_DIALECTS = frozenset({
    "builtin", "func", "arith", "math", "memref", "scf", "linalg", "affine",
    "vector", "tensor", "bufferization", "cf", "index", "complex", "emitc",
    "llvm", "gpu", "nvvm", "rocdl", "amdgpu", "spirv", "openacc", "openmp",
    "omp", "pdl", "transform", "shape", "quant", "ub", "ptr", "amx",
    "arm_neon", "arm_sve", "x86vector", "dlti", "vcix", "riscv_v", "riscv",
})

# Data dtypes accepted by the generic profile's (pending) dtype policy.  Index
# is structural and is never judged as a data dtype.
_COMMON_DTYPES = ["f16", "bf16", "f32", "f64", "i1", "i4", "i8", "i16", "i32",
                  "i64", "u8", "u16", "u32", "u64"]

GENERIC_PROFILE = {
    "schema_version": 1,
    "name": "generic",
    "version": "1",
    "status": "confirmed",
    "description": "No target contract: only standard MLIR knowledge and structural facts.",
    "allowed_dialects": sorted(STANDARD_DIALECTS),
    "forbidden_dialects": [],
    "required_dialects": [],
    "entry_symbol": None,
    "dialects": {},
    "dtype_policy": None,
    "alignment_policy": None,
    "stage_rules": {},
    "pending": [],
    "notes": ["generic 不是交付契约：它只回答“结构上是否可识别”，不回答 AME 形态是否正确。"],
}

AME_CANDIDATE_PROFILE = {
    "schema_version": 1,
    "name": "ame-candidate",
    "version": "0.1",
    "status": "candidate",
    "description": "AME-ready MLIR 的候选契约骨架；未确认项一律 pending/candidate。",
    "allowed_dialects": ["arith", "affine", "builtin", "func", "linalg", "math",
                         "memref", "scf", "tensor", "vector", "ame"],
    "forbidden_dialects": ["gpu", "nvvm", "rocdl", "amdgpu", "spirv"],
    "required_dialects": [],
    "entry_symbol": None,
    "dialects": {
        "ame": {
            "status": "candidate",
            "closed_world": True,
            "ops": {},
            "allowed_dtypes": ["f16", "bf16", "f32"],
            "attributes_required": [],
            "attributes_allowed": None,
            "note": "命名空间、操作集合、属性均待厂商口径；当前任何 ame.* 都只能算未确认。",
        },
    },
    "dtype_policy": {
        "status": "candidate",
        "allowed": _COMMON_DTYPES,
        "note": "dtype 细则（舍入/饱和/NaN）待厂商文档；这里只做集合级候选约束。",
    },
    "alignment_policy": {
        "status": "pending",
        "alloc_ops": ["memref.alloc", "memref.alloca"],
        "required_alignment_bytes": None,
        "note": "片上存储/对齐约束未确认。",
    },
    "stage_rules": {},
    "pending": [
        "AME 方言命名空间（当前按 'ame' 占位）",
        "AME 操作集合与各自的操作数/结果/region 契约",
        "AME 属性集合与必填项",
        "dtype 细则（舍入模式、饱和/NaN、转换点）",
        "存储层次与对齐要求",
        "入口符号与 ABI 类型约定",
    ],
    "notes": [
        "本 profile 是骨架，不是结论；未确认项的违反只报 uncertain。",
        "拿到厂商文档后优先修改 profile 数据，不改检查器代码。",
    ],
}

BUILTIN_PROFILES = {
    "generic": GENERIC_PROFILE,
    "ame-candidate": AME_CANDIDATE_PROFILE,
}

"""Shared operation names; detection is not a full syntax/type verifier."""
TIR_METADATA = {"T.func_attr", "T.reads", "T.writes", "T.attr", "T.target", "T.block", "T.block_attr"}
TIR_CALLS = TIR_METADATA | {"T.match_buffer", "T.alloc_buffer", "T.launch_thread", "T.region",
    "T.copy", "T.fill", "T.min", "T.max", "T.exp", "T.exp2", "T.infinity", "T.rsqrt", "T.sqrt", "T.sigmoid", "T.handle", "T.tvm_storage_sync", "T.if_then_else", "T.Cast", "T.assume", "T.reduce",
    "T.serial", "T.parallel", "T.unroll", "T.vectorized", "range"}
TIR_SCALARS = {"T." + n for n in ["int8", "uint8", "int16", "uint16", "int32", "uint32",
    "int64", "uint64", "float32", "float64", "float16", "bfloat16", "float8_e4m3fn", "bool"]}
MLIR_OPS = {"memref.alloca", "memref.alloc", "memref.load", "memref.store", "memref.dim",
    "arith.constant", "arith.index_cast", "arith.index_castui", "arith.extsi", "arith.extui", "arith.trunci", "arith.uitofp", "arith.addi", "arith.subi", "arith.muli",
    "arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.andi", "arith.ori",
    "memref.copy", "arith.negf", "arith.cmpf", "arith.floordivsi", "arith.remsi", "arith.sitofp", "math.exp", "math.exp2", "math.rsqrt", "math.sqrt", "memref.subview", "arith.cmpi", "arith.extf", "arith.truncf", "arith.select", "linalg.fill",
    "scf.for", "scf.parallel", "scf.if", "scf.yield", "scf.reduce", "return"}
LOW_PRECISION = {"bf16", "bfloat16", "f16", "float16", "f8E4M3FN", "float8_e4m3fn"}

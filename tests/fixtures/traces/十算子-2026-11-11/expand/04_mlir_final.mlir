module {
  func.func @expand_to_mhc_fwd_kernel(%arg0: memref<?x16xbf16>, %arg1: memref<?x3x16xbf16>) {
    %c3 = arith.constant 3 : index
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c16_i32 = arith.constant 16 : i32
    %cst = arith.constant 0.000000e+00 : bf16
    %c1 = arith.constant 1 : index
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %c0 = arith.constant 0 : index
    %alloca = memref.alloca() : memref<32x128xbf16>
    %dim = memref.dim %arg0, %c0 : memref<?x16xbf16>
    %0 = arith.index_cast %dim : index to i32
    %1 = arith.addi %0, %c31_i32 : i32
    %2 = arith.floordivsi %1, %c32_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    scf.for %arg2 = %c0 to %3 step %c1 {
      scf.for %arg3 = %c0 to %c32 step %c1 {
        scf.for %arg4 = %c0 to %c128 step %c1 {
          memref.store %cst, %alloca[%arg3, %arg4] : memref<32x128xbf16>
        }
      }
      scf.for %arg3 = %c0 to %c32 step %c1 {
        scf.for %arg4 = %c0 to %c128 step %c1 {
          %4 = arith.index_cast %arg2 : index to i32
          %5 = arith.muli %4, %c32_i32 : i32
          %6 = arith.index_cast %5 : i32 to index
          %7 = arith.addi %arg3, %6 : index
          %8 = arith.cmpi sge, %arg3, %c0 : index
          %9 = arith.cmpi slt, %arg3, %c32 : index
          %10 = arith.andi %8, %9 : i1
          %11 = arith.cmpi sge, %arg4, %c0 : index
          %12 = arith.andi %10, %11 : i1
          %13 = arith.cmpi slt, %arg4, %c128 : index
          %14 = arith.andi %12, %13 : i1
          scf.if %14 {
            %15 = arith.cmpi sge, %7, %c0 : index
            %16 = arith.cmpi slt, %7, %dim : index
            %17 = arith.andi %15, %16 : i1
            %18 = arith.andi %17, %11 : i1
            %19 = arith.cmpi slt, %arg4, %c16 : index
            %20 = arith.andi %18, %19 : i1
            %21 = scf.if %20 -> (bf16) {
              %22 = memref.load %arg0[%7, %arg4] : memref<?x16xbf16>
              scf.yield %22 : bf16
            } else {
              scf.yield %cst : bf16
            }
            memref.store %21, %alloca[%arg3, %arg4] : memref<32x128xbf16>
          }
        }
      }
      scf.for %arg3 = %c0 to %c3 step %c1 {
        scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c32, %c128) step (%c1, %c1) {
          %4 = arith.index_cast %arg2 : index to i32
          %5 = arith.muli %4, %c32_i32 : i32
          %6 = arith.index_cast %arg4 : index to i32
          %7 = arith.addi %5, %6 : i32
          %8 = arith.cmpi slt, %7, %0 : i32
          %9 = arith.index_cast %arg5 : index to i32
          %10 = arith.cmpi slt, %9, %c16_i32 : i32
          %11 = arith.andi %8, %10 : i1
          scf.if %11 {
            %12 = arith.index_cast %7 : i32 to index
            %13 = memref.load %alloca[%arg4, %arg5] : memref<32x128xbf16>
            memref.store %13, %arg1[%12, %arg3, %arg5] : memref<?x3x16xbf16>
          }
          scf.reduce 
        }
      }
    }
    return
  }
}


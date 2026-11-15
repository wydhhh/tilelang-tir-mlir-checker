module {
  func.func @expand_to_mhc_fwd_kernel(%arg0: memref<?x16xbf16>, %arg1: memref<?x3x16xbf16>) {
    %alloca = memref.alloca() : memref<32x128xbf16>
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?x16xbf16>
    %c0_0 = arith.constant 0 : index
    %0 = arith.index_cast %dim : index to i32
    %c31_i32 = arith.constant 31 : i32
    %1 = arith.addi %0, %c31_i32 : i32
    %c32_i32 = arith.constant 32 : i32
    %2 = arith.floordivsi %1, %c32_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %c1 = arith.constant 1 : index
    scf.for %arg2 = %c0_0 to %3 step %c1 {
      %c0_i32 = arith.constant 0 : i32
      %c0_i32_1 = arith.constant 0 : i32
      %c0_2 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      %4 = arith.index_cast %c1_i32 : i32 to index
      %c1_3 = arith.constant 1 : index
      scf.for %arg3 = %c0_2 to %4 step %c1_3 {
        %c0_4 = arith.constant 0 : index
        %c1_i32_5 = arith.constant 1 : i32
        %5 = arith.index_cast %c1_i32_5 : i32 to index
        %c1_6 = arith.constant 1 : index
        scf.for %arg4 = %c0_4 to %5 step %c1_6 {
          %cst = arith.constant 0.000000e+00 : bf16
          linalg.fill ins(%cst : bf16) outs(%alloca : memref<32x128xbf16>)
          %c0_7 = arith.constant 0 : index
          %c32_i32_8 = arith.constant 32 : i32
          %6 = arith.index_cast %c32_i32_8 : i32 to index
          %c1_9 = arith.constant 1 : index
          scf.for %arg5 = %c0_7 to %6 step %c1_9 {
            %c0_12 = arith.constant 0 : index
            %c128_i32 = arith.constant 128 : i32
            %10 = arith.index_cast %c128_i32 : i32 to index
            %c1_13 = arith.constant 1 : index
            scf.for %arg6 = %c0_12 to %10 step %c1_13 {
              %11 = arith.index_cast %arg2 : index to i32
              %c32_i32_14 = arith.constant 32 : i32
              %12 = arith.muli %11, %c32_i32_14 : i32
              %13 = arith.index_cast %12 : i32 to index
              %14 = arith.addi %arg5, %13 : index
              %true = arith.constant true
              %c32_i32_15 = arith.constant 32 : i32
              %15 = arith.index_cast %c32_i32_15 : i32 to index
              %c0_16 = arith.constant 0 : index
              %16 = arith.cmpi sge, %arg5, %c0_16 : index
              %17 = arith.andi %true, %16 : i1
              %18 = arith.cmpi slt, %arg5, %15 : index
              %19 = arith.andi %17, %18 : i1
              %c128_i32_17 = arith.constant 128 : i32
              %20 = arith.index_cast %c128_i32_17 : i32 to index
              %c0_18 = arith.constant 0 : index
              %21 = arith.cmpi sge, %arg6, %c0_18 : index
              %22 = arith.andi %19, %21 : i1
              %23 = arith.cmpi slt, %arg6, %20 : index
              %24 = arith.andi %22, %23 : i1
              scf.if %24 {
                %true_19 = arith.constant true
                %c0_20 = arith.constant 0 : index
                %25 = arith.cmpi sge, %14, %c0_20 : index
                %26 = arith.andi %true_19, %25 : i1
                %27 = arith.cmpi slt, %14, %dim : index
                %28 = arith.andi %26, %27 : i1
                %c16_i32 = arith.constant 16 : i32
                %29 = arith.index_cast %c16_i32 : i32 to index
                %c0_21 = arith.constant 0 : index
                %30 = arith.cmpi sge, %arg6, %c0_21 : index
                %31 = arith.andi %28, %30 : i1
                %32 = arith.cmpi slt, %arg6, %29 : index
                %33 = arith.andi %31, %32 : i1
                %34 = scf.if %33 -> (bf16) {
                  %35 = memref.load %arg0[%14, %arg6] : memref<?x16xbf16>
                  scf.yield %35 : bf16
                } else {
                  %cst_22 = arith.constant 0.000000e+00 : bf16
                  scf.yield %cst_22 : bf16
                }
                memref.store %34, %alloca[%arg5, %arg6] : memref<32x128xbf16>
              }
            }
          }
          %c0_i32_10 = arith.constant 0 : i32
          %7 = arith.index_cast %c0_i32_10 : i32 to index
          %c3_i32 = arith.constant 3 : i32
          %8 = arith.index_cast %c3_i32 : i32 to index
          %c1_11 = arith.constant 1 : index
          %9 = arith.addi %7, %8 : index
          scf.for %arg5 = %7 to %9 step %c1_11 {
            %c0_i32_12 = arith.constant 0 : i32
            %10 = arith.index_cast %c0_i32_12 : i32 to index
            %c32_i32_13 = arith.constant 32 : i32
            %11 = arith.index_cast %c32_i32_13 : i32 to index
            %c1_14 = arith.constant 1 : index
            %12 = arith.addi %10, %11 : index
            scf.parallel (%arg6) = (%10) to (%12) step (%c1_14) {
              %c0_i32_15 = arith.constant 0 : i32
              %13 = arith.index_cast %c0_i32_15 : i32 to index
              %c128_i32 = arith.constant 128 : i32
              %14 = arith.index_cast %c128_i32 : i32 to index
              %c1_16 = arith.constant 1 : index
              %15 = arith.addi %13, %14 : index
              scf.parallel (%arg7) = (%13) to (%15) step (%c1_16) {
                %16 = arith.index_cast %arg2 : index to i32
                %c32_i32_17 = arith.constant 32 : i32
                %17 = arith.muli %16, %c32_i32_17 : i32
                %18 = arith.index_cast %arg6 : index to i32
                %19 = arith.addi %17, %18 : i32
                %20 = arith.index_cast %dim : index to i32
                %21 = arith.cmpi slt, %19, %20 : i32
                %22 = arith.index_cast %arg7 : index to i32
                %c16_i32 = arith.constant 16 : i32
                %23 = arith.cmpi slt, %22, %c16_i32 : i32
                %24 = arith.andi %21, %23 : i1
                scf.if %24 {
                  %25 = arith.index_cast %arg2 : index to i32
                  %c32_i32_18 = arith.constant 32 : i32
                  %26 = arith.muli %25, %c32_i32_18 : i32
                  %27 = arith.index_cast %arg6 : index to i32
                  %28 = arith.addi %26, %27 : i32
                  %29 = arith.index_cast %28 : i32 to index
                  %30 = memref.load %alloca[%arg6, %arg7] : memref<32x128xbf16>
                  memref.store %30, %arg1[%29, %arg5, %arg7] : memref<?x3x16xbf16>
                }
                scf.reduce 
              }
              scf.reduce 
            }
          }
        }
      }
    }
    return
  }
}

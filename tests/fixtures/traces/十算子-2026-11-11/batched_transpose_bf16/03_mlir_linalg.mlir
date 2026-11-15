module {
  func.func @batched_transpose_kernel(%arg0: memref<?x?x?xbf16, strided<[?, ?, 1]>>, %arg1: memref<?x?x?xbf16>) {
    %alloca = memref.alloca() : memref<4xbf16>
    %alloca_0 = memref.alloca() : memref<4x4xbf16>
    %alloca_1 = memref.alloca() : memref<64x68xbf16>
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?x?x?xbf16, strided<[?, ?, 1]>>
    %c1 = arith.constant 1 : index
    %dim_2 = memref.dim %arg0, %c1 : memref<?x?x?xbf16, strided<[?, ?, 1]>>
    %c2 = arith.constant 2 : index
    %dim_3 = memref.dim %arg0, %c2 : memref<?x?x?xbf16, strided<[?, ?, 1]>>
    %c0_4 = arith.constant 0 : index
    %0 = arith.index_cast %dim_3 : index to i32
    %c64_i32 = arith.constant 64 : i32
    %1 = arith.floordivsi %0, %c64_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %c1_5 = arith.constant 1 : index
    scf.for %arg2 = %c0_4 to %2 step %c1_5 {
      %c0_6 = arith.constant 0 : index
      %3 = arith.index_cast %dim_2 : index to i32
      %c64_i32_7 = arith.constant 64 : i32
      %4 = arith.floordivsi %3, %c64_i32_7 : i32
      %5 = arith.index_cast %4 : i32 to index
      %c1_8 = arith.constant 1 : index
      scf.for %arg3 = %c0_6 to %5 step %c1_8 {
        %c0_9 = arith.constant 0 : index
        %c1_10 = arith.constant 1 : index
        scf.for %arg4 = %c0_9 to %dim step %c1_10 {
          %cst = arith.constant 0.000000e+00 : bf16
          linalg.fill ins(%cst : bf16) outs(%alloca_1 : memref<64x68xbf16>)
          %c0_i32 = arith.constant 0 : i32
          %c0_i32_11 = arith.constant 0 : i32
          %c0_12 = arith.constant 0 : index
          %c256_i32 = arith.constant 256 : i32
          %6 = arith.index_cast %c256_i32 : i32 to index
          %c1_13 = arith.constant 1 : index
          scf.for %arg5 = %c0_12 to %6 step %c1_13 {
            %c0_19 = arith.constant 0 : index
            %c1_i32 = arith.constant 1 : i32
            %8 = arith.index_cast %c1_i32 : i32 to index
            %c1_20 = arith.constant 1 : index
            scf.for %arg6 = %c0_19 to %8 step %c1_20 {
              %c0_21 = arith.constant 0 : index
              %c1_i32_22 = arith.constant 1 : i32
              %9 = arith.index_cast %c1_i32_22 : i32 to index
              %c1_23 = arith.constant 1 : index
              scf.for %arg7 = %c0_21 to %9 step %c1_23 {
                %c0_i32_24 = arith.constant 0 : i32
                %10 = arith.index_cast %c0_i32_24 : i32 to index
                %c1_i32_25 = arith.constant 1 : i32
                %11 = arith.index_cast %c1_i32_25 : i32 to index
                %c1_26 = arith.constant 1 : index
                %12 = arith.addi %10, %11 : index
                scf.for %arg8 = %10 to %12 step %c1_26 {
                  %c0_i32_27 = arith.constant 0 : i32
                  %13 = arith.index_cast %c0_i32_27 : i32 to index
                  %c4_i32 = arith.constant 4 : i32
                  %14 = arith.index_cast %c4_i32 : i32 to index
                  %c1_28 = arith.constant 1 : index
                  %15 = arith.addi %13, %14 : index
                  scf.for %arg9 = %13 to %15 step %c1_28 {
                    %c0_i32_32 = arith.constant 0 : i32
                    %19 = arith.index_cast %c0_i32_32 : i32 to index
                    %c4_i32_33 = arith.constant 4 : i32
                    %20 = arith.index_cast %c4_i32_33 : i32 to index
                    %c1_34 = arith.constant 1 : index
                    %21 = arith.addi %19, %20 : index
                    scf.for %arg10 = %19 to %21 step %c1_34 {
                      %25 = arith.index_cast %arg3 : index to i32
                      %c64_i32_38 = arith.constant 64 : i32
                      %26 = arith.muli %25, %c64_i32_38 : i32
                      %27 = arith.index_cast %arg5 : index to i32
                      %c16_i32 = arith.constant 16 : i32
                      %28 = arith.floordivsi %27, %c16_i32 : i32
                      %c4_i32_39 = arith.constant 4 : i32
                      %29 = arith.muli %28, %c4_i32_39 : i32
                      %30 = arith.addi %26, %29 : i32
                      %31 = arith.index_cast %arg9 : index to i32
                      %32 = arith.addi %30, %31 : i32
                      %33 = arith.index_cast %32 : i32 to index
                      %34 = arith.index_cast %arg2 : index to i32
                      %c64_i32_40 = arith.constant 64 : i32
                      %35 = arith.muli %34, %c64_i32_40 : i32
                      %36 = arith.index_cast %arg5 : index to i32
                      %c16_i32_41 = arith.constant 16 : i32
                      %37 = arith.floordivsi %36, %c16_i32_41 : i32
                      %38 = arith.muli %37, %c16_i32_41 : i32
                      %39 = arith.subi %36, %38 : i32
                      %c4_i32_42 = arith.constant 4 : i32
                      %40 = arith.muli %39, %c4_i32_42 : i32
                      %41 = arith.addi %35, %40 : i32
                      %42 = arith.index_cast %arg10 : index to i32
                      %43 = arith.addi %41, %42 : i32
                      %44 = arith.index_cast %43 : i32 to index
                      %45 = memref.load %arg0[%arg4, %33, %44] : memref<?x?x?xbf16, strided<[?, ?, 1]>>
                      memref.store %45, %alloca[%arg10] : memref<4xbf16>
                    }
                    %c0_i32_35 = arith.constant 0 : i32
                    %22 = arith.index_cast %c0_i32_35 : i32 to index
                    %c4_i32_36 = arith.constant 4 : i32
                    %23 = arith.index_cast %c4_i32_36 : i32 to index
                    %c1_37 = arith.constant 1 : index
                    %24 = arith.addi %22, %23 : index
                    scf.for %arg10 = %22 to %24 step %c1_37 {
                      %25 = memref.load %alloca[%arg10] : memref<4xbf16>
                      memref.store %25, %alloca_0[%arg10, %arg9] : memref<4x4xbf16>
                    }
                  }
                  %c0_i32_29 = arith.constant 0 : i32
                  %16 = arith.index_cast %c0_i32_29 : i32 to index
                  %c4_i32_30 = arith.constant 4 : i32
                  %17 = arith.index_cast %c4_i32_30 : i32 to index
                  %c1_31 = arith.constant 1 : index
                  %18 = arith.addi %16, %17 : index
                  scf.for %arg9 = %16 to %18 step %c1_31 {
                    %c0_i32_32 = arith.constant 0 : i32
                    %19 = arith.index_cast %c0_i32_32 : i32 to index
                    %c4_i32_33 = arith.constant 4 : i32
                    %20 = arith.index_cast %c4_i32_33 : i32 to index
                    %c1_34 = arith.constant 1 : index
                    %21 = arith.addi %19, %20 : index
                    scf.for %arg10 = %19 to %21 step %c1_34 {
                      %22 = arith.index_cast %arg5 : index to i32
                      %c16_i32 = arith.constant 16 : i32
                      %23 = arith.floordivsi %22, %c16_i32 : i32
                      %24 = arith.muli %23, %c16_i32 : i32
                      %25 = arith.subi %22, %24 : i32
                      %c4_i32_35 = arith.constant 4 : i32
                      %26 = arith.muli %25, %c4_i32_35 : i32
                      %27 = arith.index_cast %arg5 : index to i32
                      %c4_i32_36 = arith.constant 4 : i32
                      %28 = arith.floordivsi %27, %c4_i32_36 : i32
                      %29 = arith.index_cast %arg9 : index to i32
                      %30 = arith.addi %28, %29 : i32
                      %c4_i32_37 = arith.constant 4 : i32
                      %31 = arith.floordivsi %30, %c4_i32_37 : i32
                      %32 = arith.muli %31, %c4_i32_37 : i32
                      %33 = arith.subi %30, %32 : i32
                      %34 = arith.addi %26, %33 : i32
                      %35 = arith.index_cast %34 : i32 to index
                      %36 = arith.index_cast %arg5 : index to i32
                      %c16_i32_38 = arith.constant 16 : i32
                      %37 = arith.floordivsi %36, %c16_i32_38 : i32
                      %c4_i32_39 = arith.constant 4 : i32
                      %38 = arith.muli %37, %c4_i32_39 : i32
                      %39 = arith.index_cast %arg10 : index to i32
                      %40 = arith.addi %38, %39 : i32
                      %41 = arith.index_cast %40 : i32 to index
                      %42 = arith.index_cast %arg5 : index to i32
                      %c4_i32_40 = arith.constant 4 : i32
                      %43 = arith.floordivsi %42, %c4_i32_40 : i32
                      %44 = arith.index_cast %arg9 : index to i32
                      %45 = arith.addi %43, %44 : i32
                      %c4_i32_41 = arith.constant 4 : i32
                      %46 = arith.floordivsi %45, %c4_i32_41 : i32
                      %47 = arith.muli %46, %c4_i32_41 : i32
                      %48 = arith.subi %45, %47 : i32
                      %49 = arith.index_cast %48 : i32 to index
                      %50 = memref.load %alloca_0[%49, %arg10] : memref<4x4xbf16>
                      memref.store %50, %alloca_1[%35, %41] : memref<64x68xbf16>
                    }
                  }
                }
              }
            }
          }
          %c0_i32_14 = arith.constant 0 : i32
          %c0_i32_15 = arith.constant 0 : i32
          %c0_16 = arith.constant 0 : index
          %c256_i32_17 = arith.constant 256 : i32
          %7 = arith.index_cast %c256_i32_17 : i32 to index
          %c1_18 = arith.constant 1 : index
          scf.for %arg5 = %c0_16 to %7 step %c1_18 {
            %c0_19 = arith.constant 0 : index
            %c1_i32 = arith.constant 1 : i32
            %8 = arith.index_cast %c1_i32 : i32 to index
            %c1_20 = arith.constant 1 : index
            scf.for %arg6 = %c0_19 to %8 step %c1_20 {
              %c0_21 = arith.constant 0 : index
              %c1_i32_22 = arith.constant 1 : i32
              %9 = arith.index_cast %c1_i32_22 : i32 to index
              %c1_23 = arith.constant 1 : index
              scf.for %arg7 = %c0_21 to %9 step %c1_23 {
                %c0_i32_24 = arith.constant 0 : i32
                %10 = arith.index_cast %c0_i32_24 : i32 to index
                %c64_i32_25 = arith.constant 64 : i32
                %11 = arith.index_cast %c64_i32_25 : i32 to index
                %c1_26 = arith.constant 1 : index
                %12 = arith.addi %10, %11 : index
                scf.parallel (%arg8) = (%10) to (%12) step (%c1_26) {
                  %c0_i32_27 = arith.constant 0 : i32
                  %13 = arith.index_cast %c0_i32_27 : i32 to index
                  %c64_i32_28 = arith.constant 64 : i32
                  %14 = arith.index_cast %c64_i32_28 : i32 to index
                  %c1_29 = arith.constant 1 : index
                  %15 = arith.addi %13, %14 : index
                  scf.parallel (%arg9) = (%13) to (%15) step (%c1_29) {
                    %16 = arith.index_cast %arg2 : index to i32
                    %c64_i32_30 = arith.constant 64 : i32
                    %17 = arith.muli %16, %c64_i32_30 : i32
                    %18 = arith.index_cast %arg8 : index to i32
                    %19 = arith.addi %17, %18 : i32
                    %20 = arith.index_cast %19 : i32 to index
                    %21 = arith.index_cast %arg3 : index to i32
                    %c64_i32_31 = arith.constant 64 : i32
                    %22 = arith.muli %21, %c64_i32_31 : i32
                    %23 = arith.index_cast %arg9 : index to i32
                    %24 = arith.addi %22, %23 : i32
                    %25 = arith.index_cast %24 : i32 to index
                    %26 = memref.load %alloca_1[%arg8, %arg9] : memref<64x68xbf16>
                    memref.store %26, %arg1[%arg4, %20, %25] : memref<?x?x?xbf16>
                    scf.reduce 
                  }
                  scf.reduce 
                }
              }
            }
          }
        }
      }
    }
    return
  }
}

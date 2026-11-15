module {
  func.func @batched_transpose_kernel(%arg0: memref<?x?x?xf32, strided<[?, ?, 1]>>, %arg1: memref<?x?x?xf32>) {
    %c68 = arith.constant 68 : index
    %c64 = arith.constant 64 : index
    %c4 = arith.constant 4 : index
    %c256 = arith.constant 256 : index
    %c2_i32 = arith.constant 2 : i32
    %c16_i32 = arith.constant 16 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c64_i32 = arith.constant 64 : i32
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloca = memref.alloca() : memref<4xf32>
    %alloca_0 = memref.alloca() : memref<4x4xf32>
    %alloca_1 = memref.alloca() : memref<64x68xf32>
    %dim = memref.dim %arg0, %c0 : memref<?x?x?xf32, strided<[?, ?, 1]>>
    %dim_2 = memref.dim %arg0, %c1 : memref<?x?x?xf32, strided<[?, ?, 1]>>
    %dim_3 = memref.dim %arg0, %c2 : memref<?x?x?xf32, strided<[?, ?, 1]>>
    %0 = arith.index_cast %dim_3 : index to i32
    %1 = arith.floordivsi %0, %c64_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    scf.for %arg2 = %c0 to %2 step %c1 {
      %3 = arith.index_cast %dim_2 : index to i32
      %4 = arith.floordivsi %3, %c64_i32 : i32
      %5 = arith.index_cast %4 : i32 to index
      scf.for %arg3 = %c0 to %5 step %c1 {
        scf.for %arg4 = %c0 to %dim step %c1 {
          scf.for %arg5 = %c0 to %c64 step %c1 {
            scf.for %arg6 = %c0 to %c68 step %c1 {
              memref.store %cst, %alloca_1[%arg5, %arg6] : memref<64x68xf32>
            }
          }
          scf.for %arg5 = %c0 to %c256 step %c1 {
            scf.for %arg6 = %c0 to %c4 step %c1 {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                %6 = arith.index_cast %arg3 : index to i32
                %7 = arith.muli %6, %c64_i32 : i32
                %8 = arith.index_cast %arg5 : index to i32
                %9 = arith.floordivsi %8, %c16_i32 : i32
                %10 = arith.muli %9, %c4_i32 : i32
                %11 = arith.addi %7, %10 : i32
                %12 = arith.index_cast %arg6 : index to i32
                %13 = arith.addi %11, %12 : i32
                %14 = arith.index_cast %13 : i32 to index
                %15 = arith.index_cast %arg2 : index to i32
                %16 = arith.muli %15, %c64_i32 : i32
                %17 = arith.muli %9, %c16_i32 : i32
                %18 = arith.subi %8, %17 : i32
                %19 = arith.muli %18, %c4_i32 : i32
                %20 = arith.addi %16, %19 : i32
                %21 = arith.index_cast %arg7 : index to i32
                %22 = arith.addi %20, %21 : i32
                %23 = arith.index_cast %22 : i32 to index
                %24 = memref.load %arg0[%arg4, %14, %23] : memref<?x?x?xf32, strided<[?, ?, 1]>>
                memref.store %24, %alloca[%arg7] : memref<4xf32>
              }
              scf.for %arg7 = %c0 to %c4 step %c1 {
                %6 = memref.load %alloca[%arg7] : memref<4xf32>
                memref.store %6, %alloca_0[%arg7, %arg6] : memref<4x4xf32>
              }
            }
            scf.for %arg6 = %c0 to %c4 step %c1 {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                %6 = arith.index_cast %arg5 : index to i32
                %7 = arith.floordivsi %6, %c16_i32 : i32
                %8 = arith.muli %7, %c16_i32 : i32
                %9 = arith.subi %6, %8 : i32
                %10 = arith.muli %9, %c4_i32 : i32
                %11 = arith.floordivsi %6, %c2_i32 : i32
                %12 = arith.index_cast %arg6 : index to i32
                %13 = arith.addi %11, %12 : i32
                %14 = arith.floordivsi %13, %c4_i32 : i32
                %15 = arith.muli %14, %c4_i32 : i32
                %16 = arith.subi %13, %15 : i32
                %17 = arith.addi %10, %16 : i32
                %18 = arith.index_cast %17 : i32 to index
                %19 = arith.muli %7, %c4_i32 : i32
                %20 = arith.index_cast %arg7 : index to i32
                %21 = arith.addi %19, %20 : i32
                %22 = arith.index_cast %21 : i32 to index
                %23 = arith.index_cast %16 : i32 to index
                %24 = memref.load %alloca_0[%23, %arg7] : memref<4x4xf32>
                memref.store %24, %alloca_1[%18, %22] : memref<64x68xf32>
              }
            }
          }
          scf.for %arg5 = %c0 to %c256 step %c1 {
            scf.parallel (%arg6, %arg7) = (%c0, %c0) to (%c64, %c64) step (%c1, %c1) {
              %6 = arith.index_cast %arg2 : index to i32
              %7 = arith.muli %6, %c64_i32 : i32
              %8 = arith.index_cast %arg6 : index to i32
              %9 = arith.addi %7, %8 : i32
              %10 = arith.index_cast %9 : i32 to index
              %11 = arith.index_cast %arg3 : index to i32
              %12 = arith.muli %11, %c64_i32 : i32
              %13 = arith.index_cast %arg7 : index to i32
              %14 = arith.addi %12, %13 : i32
              %15 = arith.index_cast %14 : i32 to index
              %16 = memref.load %alloca_1[%arg6, %arg7] : memref<64x68xf32>
              memref.store %16, %arg1[%arg4, %10, %15] : memref<?x?x?xf32>
              scf.reduce 
            }
          }
        }
      }
    }
    return
  }
}


module {
  func.func @normalize_weight_kernel(%arg0: memref<?x4xf32>, %arg1: memref<?xf32>, %arg2: memref<?x4xf32>) {
    %c4 = arith.constant 4 : index
    %c128 = arith.constant 128 : index
    %cst = arith.constant 9.99999968E-21 : f32
    %c1 = arith.constant 1 : index
    %c128_i32 = arith.constant 128 : i32
    %c127_i32 = arith.constant 127 : i32
    %c0 = arith.constant 0 : index
    %alloca = memref.alloca() : memref<128xf32>
    %alloca_0 = memref.alloca() : memref<4xf32>
    %dim = memref.dim %arg0, %c0 : memref<?x4xf32>
    %0 = arith.index_cast %dim : index to i32
    %1 = arith.addi %0, %c127_i32 : i32
    %2 = arith.floordivsi %1, %c128_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    scf.for %arg3 = %c0 to %3 step %c1 {
      scf.for %arg4 = %c0 to %c128 step %c1 {
        scf.for %arg5 = %c0 to %c128 step %c1 {
          memref.store %cst, %alloca[%arg5] : memref<128xf32>
        }
        %subview = memref.subview %alloca[%arg4] [1] [1] : memref<128xf32> to memref<1xf32, strided<[1], offset: ?>>
        %4 = arith.index_cast %arg3 : index to i32
        %5 = arith.muli %4, %c128_i32 : i32
        %6 = arith.index_cast %arg4 : index to i32
        %7 = arith.addi %5, %6 : i32
        %8 = arith.cmpi slt, %7, %0 : i32
        scf.if %8 {
          scf.for %arg5 = %c0 to %c4 step %c1 {
            %11 = arith.index_cast %7 : i32 to index
            %12 = memref.load %arg0[%11, %arg5] : memref<?x4xf32>
            memref.store %12, %alloca_0[%arg5] : memref<4xf32>
          }
          scf.for %arg5 = %c0 to %c4 step %c1 {
            %11 = memref.load %subview[%c0] : memref<1xf32, strided<[1], offset: ?>>
            %12 = memref.load %alloca_0[%arg5] : memref<4xf32>
            %13 = arith.addf %11, %12 : f32
            memref.store %13, %subview[%c0] : memref<1xf32, strided<[1], offset: ?>>
          }
          %9 = arith.index_cast %7 : i32 to index
          %10 = memref.load %subview[%c0] : memref<1xf32, strided<[1], offset: ?>>
          memref.store %10, %arg1[%9] : memref<?xf32>
          scf.for %arg5 = %c0 to %c4 step %c1 {
            %11 = memref.load %alloca_0[%arg5] : memref<4xf32>
            %12 = memref.load %subview[%c0] : memref<1xf32, strided<[1], offset: ?>>
            %13 = arith.divf %11, %12 : f32
            memref.store %13, %arg2[%9, %arg5] : memref<?x4xf32>
          }
        }
      }
    }
    return
  }
}


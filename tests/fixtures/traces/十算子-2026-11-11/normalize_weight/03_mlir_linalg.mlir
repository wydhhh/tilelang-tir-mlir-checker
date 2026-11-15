module {
  func.func @normalize_weight_kernel(%arg0: memref<?x4xf32>, %arg1: memref<?xf32>, %arg2: memref<?x4xf32>) {
    %alloca = memref.alloca() : memref<128xf32>
    %alloca_0 = memref.alloca() : memref<4xf32>
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?x4xf32>
    %c0_1 = arith.constant 0 : index
    %0 = arith.index_cast %dim : index to i32
    %c127_i32 = arith.constant 127 : i32
    %1 = arith.addi %0, %c127_i32 : i32
    %c128_i32 = arith.constant 128 : i32
    %2 = arith.floordivsi %1, %c128_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0_1 to %3 step %c1 {
      %c0_i32 = arith.constant 0 : i32
      %c0_i32_2 = arith.constant 0 : i32
      %c0_3 = arith.constant 0 : index
      %c128_i32_4 = arith.constant 128 : i32
      %4 = arith.index_cast %c128_i32_4 : i32 to index
      %c1_5 = arith.constant 1 : index
      scf.for %arg4 = %c0_3 to %4 step %c1_5 {
        %c0_6 = arith.constant 0 : index
        %c1_i32 = arith.constant 1 : i32
        %5 = arith.index_cast %c1_i32 : i32 to index
        %c1_7 = arith.constant 1 : index
        scf.for %arg5 = %c0_6 to %5 step %c1_7 {
          %c0_8 = arith.constant 0 : index
          %c1_i32_9 = arith.constant 1 : i32
          %6 = arith.index_cast %c1_i32_9 : i32 to index
          %c1_10 = arith.constant 1 : index
          scf.for %arg6 = %c0_8 to %6 step %c1_10 {
            %c0_11 = arith.constant 0 : index
            %c128_i32_12 = arith.constant 128 : i32
            %7 = arith.index_cast %c128_i32_12 : i32 to index
            %c1_13 = arith.constant 1 : index
            scf.for %arg7 = %c0_11 to %7 step %c1_13 {
              %cst = arith.constant 9.99999968E-21 : f32
              memref.store %cst, %alloca[%arg7] : memref<128xf32>
            }
            %subview = memref.subview %alloca[%arg4] [1] [1] : memref<128xf32> to memref<1xf32, strided<[1], offset: ?>>
            %8 = arith.index_cast %arg3 : index to i32
            %c128_i32_14 = arith.constant 128 : i32
            %9 = arith.muli %8, %c128_i32_14 : i32
            %10 = arith.index_cast %arg4 : index to i32
            %11 = arith.addi %9, %10 : i32
            %12 = arith.index_cast %dim : index to i32
            %13 = arith.cmpi slt, %11, %12 : i32
            scf.if %13 {
              %c0_i32_15 = arith.constant 0 : i32
              %14 = arith.index_cast %c0_i32_15 : i32 to index
              %c4_i32 = arith.constant 4 : i32
              %15 = arith.index_cast %c4_i32 : i32 to index
              %c1_16 = arith.constant 1 : index
              %16 = arith.addi %14, %15 : index
              scf.for %arg7 = %14 to %16 step %c1_16 {
                %30 = arith.index_cast %arg3 : index to i32
                %c128_i32_25 = arith.constant 128 : i32
                %31 = arith.muli %30, %c128_i32_25 : i32
                %32 = arith.index_cast %arg4 : index to i32
                %33 = arith.addi %31, %32 : i32
                %34 = arith.index_cast %33 : i32 to index
                %35 = memref.load %arg0[%34, %arg7] : memref<?x4xf32>
                memref.store %35, %alloca_0[%arg7] : memref<4xf32>
              }
              %c0_i32_17 = arith.constant 0 : i32
              %17 = arith.index_cast %c0_i32_17 : i32 to index
              %c4_i32_18 = arith.constant 4 : i32
              %18 = arith.index_cast %c4_i32_18 : i32 to index
              %c1_19 = arith.constant 1 : index
              %19 = arith.addi %17, %18 : index
              scf.for %arg7 = %17 to %19 step %c1_19 {
                %c0_i32_25 = arith.constant 0 : i32
                %30 = arith.index_cast %c0_i32_25 : i32 to index
                %c0_i32_26 = arith.constant 0 : i32
                %31 = arith.index_cast %c0_i32_26 : i32 to index
                %32 = memref.load %subview[%31] : memref<1xf32, strided<[1], offset: ?>>
                %33 = memref.load %alloca_0[%arg7] : memref<4xf32>
                %34 = arith.addf %32, %33 : f32
                memref.store %34, %subview[%30] : memref<1xf32, strided<[1], offset: ?>>
              }
              %20 = arith.index_cast %arg3 : index to i32
              %c128_i32_20 = arith.constant 128 : i32
              %21 = arith.muli %20, %c128_i32_20 : i32
              %22 = arith.index_cast %arg4 : index to i32
              %23 = arith.addi %21, %22 : i32
              %24 = arith.index_cast %23 : i32 to index
              %c0_i32_21 = arith.constant 0 : i32
              %25 = arith.index_cast %c0_i32_21 : i32 to index
              %26 = memref.load %subview[%25] : memref<1xf32, strided<[1], offset: ?>>
              memref.store %26, %arg1[%24] : memref<?xf32>
              %c0_i32_22 = arith.constant 0 : i32
              %27 = arith.index_cast %c0_i32_22 : i32 to index
              %c4_i32_23 = arith.constant 4 : i32
              %28 = arith.index_cast %c4_i32_23 : i32 to index
              %c1_24 = arith.constant 1 : index
              %29 = arith.addi %27, %28 : index
              scf.for %arg7 = %27 to %29 step %c1_24 {
                %30 = arith.index_cast %arg3 : index to i32
                %c128_i32_25 = arith.constant 128 : i32
                %31 = arith.muli %30, %c128_i32_25 : i32
                %32 = arith.index_cast %arg4 : index to i32
                %33 = arith.addi %31, %32 : i32
                %34 = arith.index_cast %33 : i32 to index
                %35 = memref.load %alloca_0[%arg7] : memref<4xf32>
                %c0_i32_26 = arith.constant 0 : i32
                %36 = arith.index_cast %c0_i32_26 : i32 to index
                %37 = memref.load %subview[%36] : memref<1xf32, strided<[1], offset: ?>>
                %38 = arith.divf %35, %37 : f32
                memref.store %38, %arg2[%34, %arg7] : memref<?x4xf32>
              }
            }
          }
        }
      }
    }
    return
  }
}

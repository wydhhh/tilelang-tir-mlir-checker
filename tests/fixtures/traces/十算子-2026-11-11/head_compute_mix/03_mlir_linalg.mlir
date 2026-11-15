module {
  func.func @mhc_head_compute_mix_fwd_kernel(%arg0: memref<?x4xf32>, %arg1: memref<1xf32>, %arg2: memref<4xf32>, %arg3: memref<?x4xf32>) {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?x4xf32>
    %c0_0 = arith.constant 0 : index
    %0 = arith.index_cast %dim : index to i32
    %c1_i32 = arith.constant 1 : i32
    %1 = arith.addi %0, %c1_i32 : i32
    %c2_i32 = arith.constant 2 : i32
    %2 = arith.floordivsi %1, %c2_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %c1 = arith.constant 1 : index
    scf.for %arg4 = %c0_0 to %3 step %c1 {
      %c0_i32 = arith.constant 0 : i32
      %c0_1 = arith.constant 0 : index
      %c1_i32_2 = arith.constant 1 : i32
      %4 = arith.index_cast %c1_i32_2 : i32 to index
      %c1_3 = arith.constant 1 : index
      scf.for %arg5 = %c0_1 to %4 step %c1_3 {
        %c0_4 = arith.constant 0 : index
        %c1_i32_5 = arith.constant 1 : i32
        %5 = arith.index_cast %c1_i32_5 : i32 to index
        %c1_6 = arith.constant 1 : index
        scf.for %arg6 = %c0_4 to %5 step %c1_6 {
          %c0_i32_7 = arith.constant 0 : i32
          %6 = arith.index_cast %c0_i32_7 : i32 to index
          %c2_i32_8 = arith.constant 2 : i32
          %7 = arith.index_cast %c2_i32_8 : i32 to index
          %c1_9 = arith.constant 1 : index
          %8 = arith.addi %6, %7 : index
          scf.parallel (%arg7) = (%6) to (%8) step (%c1_9) {
            %c0_i32_10 = arith.constant 0 : i32
            %9 = arith.index_cast %c0_i32_10 : i32 to index
            %c4_i32 = arith.constant 4 : i32
            %10 = arith.index_cast %c4_i32 : i32 to index
            %c1_11 = arith.constant 1 : index
            %11 = arith.addi %9, %10 : index
            scf.parallel (%arg8) = (%9) to (%11) step (%c1_11) {
              %12 = arith.index_cast %arg4 : index to i32
              %c2_i32_12 = arith.constant 2 : i32
              %13 = arith.muli %12, %c2_i32_12 : i32
              %14 = arith.index_cast %arg7 : index to i32
              %15 = arith.addi %13, %14 : i32
              %16 = arith.index_cast %dim : index to i32
              %17 = arith.cmpi slt, %15, %16 : i32
              scf.if %17 {
                %18 = arith.index_cast %arg4 : index to i32
                %c2_i32_13 = arith.constant 2 : i32
                %19 = arith.muli %18, %c2_i32_13 : i32
                %20 = arith.index_cast %arg7 : index to i32
                %21 = arith.addi %19, %20 : i32
                %22 = arith.index_cast %21 : i32 to index
                %23 = arith.index_cast %arg4 : index to i32
                %c2_i32_14 = arith.constant 2 : i32
                %24 = arith.muli %23, %c2_i32_14 : i32
                %25 = arith.index_cast %arg7 : index to i32
                %26 = arith.addi %24, %25 : i32
                %27 = arith.index_cast %26 : i32 to index
                %28 = memref.load %arg0[%27, %arg8] : memref<?x4xf32>
                %c0_i32_15 = arith.constant 0 : i32
                %29 = arith.index_cast %c0_i32_15 : i32 to index
                %30 = memref.load %arg1[%29] : memref<1xf32>
                %31 = arith.mulf %28, %30 : f32
                %32 = memref.load %arg2[%arg8] : memref<4xf32>
                %33 = arith.addf %31, %32 : f32
                %cst = arith.constant 0.000000e+00 : f32
                %cst_16 = arith.constant 1.000000e+00 : f32
                %34 = arith.subf %cst, %33 : f32
                %35 = math.exp %34 : f32
                %36 = arith.addf %cst_16, %35 : f32
                %37 = arith.divf %cst_16, %36 : f32
                %cst_17 = arith.constant 9.99999997E-7 : f32
                %38 = arith.addf %37, %cst_17 : f32
                memref.store %38, %arg3[%22, %arg8] : memref<?x4xf32>
              }
              scf.reduce 
            }
            scf.reduce 
          }
        }
      }
    }
    return
  }
}

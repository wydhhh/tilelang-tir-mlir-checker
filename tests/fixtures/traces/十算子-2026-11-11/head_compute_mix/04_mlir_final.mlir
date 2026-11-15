module {
  func.func @mhc_head_compute_mix_fwd_kernel(%arg0: memref<?x4xf32>, %arg1: memref<1xf32>, %arg2: memref<4xf32>, %arg3: memref<?x4xf32>) {
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 9.99999997E-7 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?x4xf32>
    %0 = arith.index_cast %dim : index to i32
    %1 = arith.addi %0, %c1_i32 : i32
    %2 = arith.floordivsi %1, %c2_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    scf.for %arg4 = %c0 to %3 step %c1 {
      scf.parallel (%arg5, %arg6) = (%c0, %c0) to (%c2, %c4) step (%c1, %c1) {
        %4 = arith.index_cast %arg4 : index to i32
        %5 = arith.muli %4, %c2_i32 : i32
        %6 = arith.index_cast %arg5 : index to i32
        %7 = arith.addi %5, %6 : i32
        %8 = arith.cmpi slt, %7, %0 : i32
        scf.if %8 {
          %9 = arith.index_cast %7 : i32 to index
          %10 = memref.load %arg0[%9, %arg6] : memref<?x4xf32>
          %11 = memref.load %arg1[%c0] : memref<1xf32>
          %12 = arith.mulf %10, %11 : f32
          %13 = memref.load %arg2[%arg6] : memref<4xf32>
          %14 = arith.addf %12, %13 : f32
          %15 = arith.subf %cst_1, %14 : f32
          %16 = math.exp %15 : f32
          %17 = arith.addf %16, %cst_0 : f32
          %18 = arith.divf %cst_0, %17 : f32
          %19 = arith.addf %18, %cst : f32
          memref.store %19, %arg3[%9, %arg6] : memref<?x4xf32>
        }
        scf.reduce 
      }
    }
    return
  }
}


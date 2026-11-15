module {
  func.func @main(%arg0: memref<35xi8>, %arg1: memref<35xf32>, %arg2: memref<35xf32>, %arg3: memref<35xf32>) {
    %alloca = memref.alloca() : memref<32xf32>
    %alloca_0 = memref.alloca() : memref<32xf32>
    %alloca_1 = memref.alloca() : memref<32xi8>
    %c0 = arith.constant 0 : index
    %c2_i32 = arith.constant 2 : i32
    %0 = arith.index_cast %c2_i32 : i32 to index
    %c1 = arith.constant 1 : index
    scf.for %arg4 = %c0 to %0 step %c1 {
      %c0_i32 = arith.constant 0 : i32
      %c0_2 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      %1 = arith.index_cast %c1_i32 : i32 to index
      %c1_3 = arith.constant 1 : index
      scf.for %arg5 = %c0_2 to %1 step %c1_3 {
        %c0_4 = arith.constant 0 : index
        %c1_i32_5 = arith.constant 1 : i32
        %2 = arith.index_cast %c1_i32_5 : i32 to index
        %c1_6 = arith.constant 1 : index
        scf.for %arg6 = %c0_4 to %2 step %c1_6 {
          %c0_i8 = arith.constant 0 : i8
          linalg.fill ins(%c0_i8 : i8) outs(%alloca_1 : memref<32xi8>)
          %cst = arith.constant 0.000000e+00 : f32
          linalg.fill ins(%cst : f32) outs(%alloca_0 : memref<32xf32>)
          %cst_7 = arith.constant 0.000000e+00 : f32
          linalg.fill ins(%cst_7 : f32) outs(%alloca : memref<32xf32>)
          %c0_8 = arith.constant 0 : index
          %c32_i32 = arith.constant 32 : i32
          %3 = arith.index_cast %c32_i32 : i32 to index
          %c1_9 = arith.constant 1 : index
          scf.for %arg7 = %c0_8 to %3 step %c1_9 {
            %10 = arith.index_cast %arg4 : index to i32
            %c32_i32_21 = arith.constant 32 : i32
            %11 = arith.muli %10, %c32_i32_21 : i32
            %12 = arith.index_cast %11 : i32 to index
            %13 = arith.addi %arg7, %12 : index
            %true = arith.constant true
            %c32_i32_22 = arith.constant 32 : i32
            %14 = arith.index_cast %c32_i32_22 : i32 to index
            %c0_23 = arith.constant 0 : index
            %15 = arith.cmpi sge, %arg7, %c0_23 : index
            %16 = arith.andi %true, %15 : i1
            %17 = arith.cmpi slt, %arg7, %14 : index
            %18 = arith.andi %16, %17 : i1
            scf.if %18 {
              %true_24 = arith.constant true
              %c35_i32 = arith.constant 35 : i32
              %19 = arith.index_cast %c35_i32 : i32 to index
              %c0_25 = arith.constant 0 : index
              %20 = arith.cmpi sge, %13, %c0_25 : index
              %21 = arith.andi %true_24, %20 : i1
              %22 = arith.cmpi slt, %13, %19 : index
              %23 = arith.andi %21, %22 : i1
              %24 = scf.if %23 -> (i8) {
                %25 = memref.load %arg0[%13] : memref<35xi8>
                scf.yield %25 : i8
              } else {
                %c0_i8_26 = arith.constant 0 : i8
                scf.yield %c0_i8_26 : i8
              }
              memref.store %24, %alloca_1[%arg7] : memref<32xi8>
            }
          }
          %c0_10 = arith.constant 0 : index
          %c32_i32_11 = arith.constant 32 : i32
          %4 = arith.index_cast %c32_i32_11 : i32 to index
          %c1_12 = arith.constant 1 : index
          scf.for %arg7 = %c0_10 to %4 step %c1_12 {
            %10 = arith.index_cast %arg4 : index to i32
            %c32_i32_21 = arith.constant 32 : i32
            %11 = arith.muli %10, %c32_i32_21 : i32
            %12 = arith.index_cast %11 : i32 to index
            %13 = arith.addi %arg7, %12 : index
            %true = arith.constant true
            %c32_i32_22 = arith.constant 32 : i32
            %14 = arith.index_cast %c32_i32_22 : i32 to index
            %c0_23 = arith.constant 0 : index
            %15 = arith.cmpi sge, %arg7, %c0_23 : index
            %16 = arith.andi %true, %15 : i1
            %17 = arith.cmpi slt, %arg7, %14 : index
            %18 = arith.andi %16, %17 : i1
            scf.if %18 {
              %true_24 = arith.constant true
              %c35_i32 = arith.constant 35 : i32
              %19 = arith.index_cast %c35_i32 : i32 to index
              %c0_25 = arith.constant 0 : index
              %20 = arith.cmpi sge, %13, %c0_25 : index
              %21 = arith.andi %true_24, %20 : i1
              %22 = arith.cmpi slt, %13, %19 : index
              %23 = arith.andi %21, %22 : i1
              %24 = scf.if %23 -> (f32) {
                %25 = memref.load %arg1[%13] : memref<35xf32>
                scf.yield %25 : f32
              } else {
                %cst_26 = arith.constant 0.000000e+00 : f32
                scf.yield %cst_26 : f32
              }
              memref.store %24, %alloca_0[%arg7] : memref<32xf32>
            }
          }
          %c0_13 = arith.constant 0 : index
          %c32_i32_14 = arith.constant 32 : i32
          %5 = arith.index_cast %c32_i32_14 : i32 to index
          %c1_15 = arith.constant 1 : index
          scf.for %arg7 = %c0_13 to %5 step %c1_15 {
            %10 = arith.index_cast %arg4 : index to i32
            %c32_i32_21 = arith.constant 32 : i32
            %11 = arith.muli %10, %c32_i32_21 : i32
            %12 = arith.index_cast %11 : i32 to index
            %13 = arith.addi %arg7, %12 : index
            %true = arith.constant true
            %c32_i32_22 = arith.constant 32 : i32
            %14 = arith.index_cast %c32_i32_22 : i32 to index
            %c0_23 = arith.constant 0 : index
            %15 = arith.cmpi sge, %arg7, %c0_23 : index
            %16 = arith.andi %true, %15 : i1
            %17 = arith.cmpi slt, %arg7, %14 : index
            %18 = arith.andi %16, %17 : i1
            scf.if %18 {
              %true_24 = arith.constant true
              %c35_i32 = arith.constant 35 : i32
              %19 = arith.index_cast %c35_i32 : i32 to index
              %c0_25 = arith.constant 0 : index
              %20 = arith.cmpi sge, %13, %c0_25 : index
              %21 = arith.andi %true_24, %20 : i1
              %22 = arith.cmpi slt, %13, %19 : index
              %23 = arith.andi %21, %22 : i1
              %24 = scf.if %23 -> (f32) {
                %25 = memref.load %arg2[%13] : memref<35xf32>
                scf.yield %25 : f32
              } else {
                %cst_26 = arith.constant 0.000000e+00 : f32
                scf.yield %cst_26 : f32
              }
              memref.store %24, %alloca[%arg7] : memref<32xf32>
            }
          }
          %c0_i32_16 = arith.constant 0 : i32
          %6 = arith.index_cast %c0_i32_16 : i32 to index
          %c8_i32 = arith.constant 8 : i32
          %7 = arith.index_cast %c8_i32 : i32 to index
          %c1_17 = arith.constant 1 : index
          %8 = arith.addi %6, %7 : index
          scf.parallel (%arg7) = (%6) to (%8) step (%c1_17) {
            %c0_i32_21 = arith.constant 0 : i32
            %10 = arith.index_cast %c0_i32_21 : i32 to index
            %c4_i32 = arith.constant 4 : i32
            %11 = arith.index_cast %c4_i32 : i32 to index
            %c1_22 = arith.constant 1 : index
            %12 = arith.addi %10, %11 : index
            scf.parallel (%arg8) = (%10) to (%12) step (%c1_22) {
              %13 = arith.index_cast %arg7 : index to i32
              %c4_i32_23 = arith.constant 4 : i32
              %14 = arith.muli %13, %c4_i32_23 : i32
              %15 = arith.index_cast %arg8 : index to i32
              %16 = arith.addi %14, %15 : i32
              %17 = arith.index_cast %16 : i32 to index
              %18 = arith.index_cast %arg7 : index to i32
              %c4_i32_24 = arith.constant 4 : i32
              %19 = arith.muli %18, %c4_i32_24 : i32
              %20 = arith.index_cast %arg8 : index to i32
              %21 = arith.addi %19, %20 : i32
              %22 = arith.index_cast %21 : i32 to index
              %23 = memref.load %alloca_1[%22] : memref<32xi8>
              %c0_i8_25 = arith.constant 0 : i8
              %24 = arith.cmpi ne, %23, %c0_i8_25 : i8
              %25 = scf.if %24 -> (f32) {
                %26 = arith.index_cast %arg7 : index to i32
                %c4_i32_26 = arith.constant 4 : i32
                %27 = arith.muli %26, %c4_i32_26 : i32
                %28 = arith.index_cast %arg8 : index to i32
                %29 = arith.addi %27, %28 : i32
                %30 = arith.index_cast %29 : i32 to index
                %31 = memref.load %alloca_0[%30] : memref<32xf32>
                scf.yield %31 : f32
              } else {
                %26 = arith.index_cast %arg7 : index to i32
                %c4_i32_26 = arith.constant 4 : i32
                %27 = arith.muli %26, %c4_i32_26 : i32
                %28 = arith.index_cast %arg8 : index to i32
                %29 = arith.addi %27, %28 : i32
                %30 = arith.index_cast %29 : i32 to index
                %31 = memref.load %alloca[%30] : memref<32xf32>
                scf.yield %31 : f32
              }
              memref.store %25, %alloca_0[%17] : memref<32xf32>
              scf.reduce 
            }
            scf.reduce 
          }
          %c0_18 = arith.constant 0 : index
          %c32_i32_19 = arith.constant 32 : i32
          %9 = arith.index_cast %c32_i32_19 : i32 to index
          %c1_20 = arith.constant 1 : index
          scf.for %arg7 = %c0_18 to %9 step %c1_20 {
            %10 = arith.index_cast %arg4 : index to i32
            %c32_i32_21 = arith.constant 32 : i32
            %11 = arith.muli %10, %c32_i32_21 : i32
            %12 = arith.index_cast %11 : i32 to index
            %13 = arith.addi %arg7, %12 : index
            %true = arith.constant true
            %c35_i32 = arith.constant 35 : i32
            %14 = arith.index_cast %c35_i32 : i32 to index
            %c0_22 = arith.constant 0 : index
            %15 = arith.cmpi sge, %13, %c0_22 : index
            %16 = arith.andi %true, %15 : i1
            %17 = arith.cmpi slt, %13, %14 : index
            %18 = arith.andi %16, %17 : i1
            scf.if %18 {
              %true_23 = arith.constant true
              %c32_i32_24 = arith.constant 32 : i32
              %19 = arith.index_cast %c32_i32_24 : i32 to index
              %c0_25 = arith.constant 0 : index
              %20 = arith.cmpi sge, %arg7, %c0_25 : index
              %21 = arith.andi %true_23, %20 : i1
              %22 = arith.cmpi slt, %arg7, %19 : index
              %23 = arith.andi %21, %22 : i1
              %24 = scf.if %23 -> (f32) {
                %25 = memref.load %alloca_0[%arg7] : memref<32xf32>
                scf.yield %25 : f32
              } else {
                %cst_26 = arith.constant 0.000000e+00 : f32
                scf.yield %cst_26 : f32
              }
              memref.store %24, %arg3[%13] : memref<35xf32>
            }
          }
        }
      }
    }
    return
  }
}

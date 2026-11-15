module {
  func.func @_mhc_pre_apply_mix_fwd_kernel(%arg0: memref<?x4x8xbf16>, %arg1: memref<?x4xf32>, %arg2: memref<?x8xbf16>) {
    %alloca = memref.alloca() : memref<8xf32>
    %alloca_0 = memref.alloca() : memref<4x8xf32>
    %alloca_1 = memref.alloca() : memref<4xf32>
    %alloca_2 = memref.alloca() : memref<8xbf16>
    %alloca_3 = memref.alloca() : memref<4x8xbf16>
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?x4x8xbf16>
    %c0_4 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0_4 to %dim step %c1 {
      %cst = arith.constant 0.000000e+00 : bf16
      linalg.fill ins(%cst : bf16) outs(%alloca_3 : memref<4x8xbf16>)
      %cst_5 = arith.constant 0.000000e+00 : bf16
      linalg.fill ins(%cst_5 : bf16) outs(%alloca_2 : memref<8xbf16>)
      %c0_i32 = arith.constant 0 : i32
      %c0_6 = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      %0 = arith.index_cast %c1_i32 : i32 to index
      %c1_7 = arith.constant 1 : index
      scf.for %arg4 = %c0_6 to %0 step %c1_7 {
        %c0_8 = arith.constant 0 : index
        %c1_i32_9 = arith.constant 1 : i32
        %1 = arith.index_cast %c1_i32_9 : i32 to index
        %c1_10 = arith.constant 1 : index
        scf.for %arg5 = %c0_8 to %1 step %c1_10 {
          %cst_11 = arith.constant 0.000000e+00 : f32
          linalg.fill ins(%cst_11 : f32) outs(%alloca_1 : memref<4xf32>)
          %cst_12 = arith.constant 0.000000e+00 : f32
          linalg.fill ins(%cst_12 : f32) outs(%alloca_0 : memref<4x8xf32>)
          %cst_13 = arith.constant 0.000000e+00 : f32
          linalg.fill ins(%cst_13 : f32) outs(%alloca : memref<8xf32>)
          %c0_14 = arith.constant 0 : index
          %c4_i32 = arith.constant 4 : i32
          %2 = arith.index_cast %c4_i32 : i32 to index
          %c1_15 = arith.constant 1 : index
          scf.for %arg6 = %c0_14 to %2 step %c1_15 {
            %true = arith.constant true
            %c4_i32_19 = arith.constant 4 : i32
            %6 = arith.index_cast %c4_i32_19 : i32 to index
            %c0_20 = arith.constant 0 : index
            %7 = arith.cmpi sge, %arg6, %c0_20 : index
            %8 = arith.andi %true, %7 : i1
            %9 = arith.cmpi slt, %arg6, %6 : index
            %10 = arith.andi %8, %9 : i1
            scf.if %10 {
              %true_21 = arith.constant true
              %c0_22 = arith.constant 0 : index
              %11 = arith.cmpi sge, %arg3, %c0_22 : index
              %12 = arith.andi %true_21, %11 : i1
              %13 = arith.cmpi slt, %arg3, %dim : index
              %14 = arith.andi %12, %13 : i1
              %c4_i32_23 = arith.constant 4 : i32
              %15 = arith.index_cast %c4_i32_23 : i32 to index
              %c0_24 = arith.constant 0 : index
              %16 = arith.cmpi sge, %arg6, %c0_24 : index
              %17 = arith.andi %14, %16 : i1
              %18 = arith.cmpi slt, %arg6, %15 : index
              %19 = arith.andi %17, %18 : i1
              %20 = scf.if %19 -> (f32) {
                %21 = memref.load %arg1[%arg3, %arg6] : memref<?x4xf32>
                scf.yield %21 : f32
              } else {
                %cst_25 = arith.constant 0.000000e+00 : f32
                scf.yield %cst_25 : f32
              }
              memref.store %20, %alloca_1[%arg6] : memref<4xf32>
            }
          }
          %c0_i32_16 = arith.constant 0 : i32
          %3 = arith.index_cast %c0_i32_16 : i32 to index
          %c1_i32_17 = arith.constant 1 : i32
          %4 = arith.index_cast %c1_i32_17 : i32 to index
          %c1_18 = arith.constant 1 : index
          %5 = arith.addi %3, %4 : index
          scf.for %arg6 = %3 to %5 step %c1_18 {
            %c0_19 = arith.constant 0 : index
            %c4_i32_20 = arith.constant 4 : i32
            %6 = arith.index_cast %c4_i32_20 : i32 to index
            %c1_21 = arith.constant 1 : index
            scf.for %arg7 = %c0_19 to %6 step %c1_21 {
              %c0_37 = arith.constant 0 : index
              %c8_i32_38 = arith.constant 8 : i32
              %14 = arith.index_cast %c8_i32_38 : i32 to index
              %c1_39 = arith.constant 1 : index
              scf.for %arg8 = %c0_37 to %14 step %c1_39 {
                %true = arith.constant true
                %c4_i32_40 = arith.constant 4 : i32
                %15 = arith.index_cast %c4_i32_40 : i32 to index
                %c0_41 = arith.constant 0 : index
                %16 = arith.cmpi sge, %arg7, %c0_41 : index
                %17 = arith.andi %true, %16 : i1
                %18 = arith.cmpi slt, %arg7, %15 : index
                %19 = arith.andi %17, %18 : i1
                %c8_i32_42 = arith.constant 8 : i32
                %20 = arith.index_cast %c8_i32_42 : i32 to index
                %c0_43 = arith.constant 0 : index
                %21 = arith.cmpi sge, %arg8, %c0_43 : index
                %22 = arith.andi %19, %21 : i1
                %23 = arith.cmpi slt, %arg8, %20 : index
                %24 = arith.andi %22, %23 : i1
                scf.if %24 {
                  %true_44 = arith.constant true
                  %c0_45 = arith.constant 0 : index
                  %25 = arith.cmpi sge, %arg3, %c0_45 : index
                  %26 = arith.andi %true_44, %25 : i1
                  %27 = arith.cmpi slt, %arg3, %dim : index
                  %28 = arith.andi %26, %27 : i1
                  %c4_i32_46 = arith.constant 4 : i32
                  %29 = arith.index_cast %c4_i32_46 : i32 to index
                  %c0_47 = arith.constant 0 : index
                  %30 = arith.cmpi sge, %arg7, %c0_47 : index
                  %31 = arith.andi %28, %30 : i1
                  %32 = arith.cmpi slt, %arg7, %29 : index
                  %33 = arith.andi %31, %32 : i1
                  %c8_i32_48 = arith.constant 8 : i32
                  %34 = arith.index_cast %c8_i32_48 : i32 to index
                  %c0_49 = arith.constant 0 : index
                  %35 = arith.cmpi sge, %arg8, %c0_49 : index
                  %36 = arith.andi %33, %35 : i1
                  %37 = arith.cmpi slt, %arg8, %34 : index
                  %38 = arith.andi %36, %37 : i1
                  %39 = scf.if %38 -> (bf16) {
                    %40 = memref.load %arg0[%arg3, %arg7, %arg8] : memref<?x4x8xbf16>
                    scf.yield %40 : bf16
                  } else {
                    %cst_50 = arith.constant 0.000000e+00 : bf16
                    scf.yield %cst_50 : bf16
                  }
                  memref.store %39, %alloca_3[%arg7, %arg8] : memref<4x8xbf16>
                }
              }
            }
            %c0_22 = arith.constant 0 : index
            %c4_i32_23 = arith.constant 4 : i32
            %7 = arith.index_cast %c4_i32_23 : i32 to index
            %c1_24 = arith.constant 1 : index
            scf.for %arg7 = %c0_22 to %7 step %c1_24 {
              %c0_37 = arith.constant 0 : index
              %c8_i32_38 = arith.constant 8 : i32
              %14 = arith.index_cast %c8_i32_38 : i32 to index
              %c1_39 = arith.constant 1 : index
              scf.for %arg8 = %c0_37 to %14 step %c1_39 {
                %true = arith.constant true
                %c4_i32_40 = arith.constant 4 : i32
                %15 = arith.index_cast %c4_i32_40 : i32 to index
                %c0_41 = arith.constant 0 : index
                %16 = arith.cmpi sge, %arg7, %c0_41 : index
                %17 = arith.andi %true, %16 : i1
                %18 = arith.cmpi slt, %arg7, %15 : index
                %19 = arith.andi %17, %18 : i1
                %c8_i32_42 = arith.constant 8 : i32
                %20 = arith.index_cast %c8_i32_42 : i32 to index
                %c0_43 = arith.constant 0 : index
                %21 = arith.cmpi sge, %arg8, %c0_43 : index
                %22 = arith.andi %19, %21 : i1
                %23 = arith.cmpi slt, %arg8, %20 : index
                %24 = arith.andi %22, %23 : i1
                scf.if %24 {
                  %true_44 = arith.constant true
                  %c4_i32_45 = arith.constant 4 : i32
                  %25 = arith.index_cast %c4_i32_45 : i32 to index
                  %c0_46 = arith.constant 0 : index
                  %26 = arith.cmpi sge, %arg7, %c0_46 : index
                  %27 = arith.andi %true_44, %26 : i1
                  %28 = arith.cmpi slt, %arg7, %25 : index
                  %29 = arith.andi %27, %28 : i1
                  %c8_i32_47 = arith.constant 8 : i32
                  %30 = arith.index_cast %c8_i32_47 : i32 to index
                  %c0_48 = arith.constant 0 : index
                  %31 = arith.cmpi sge, %arg8, %c0_48 : index
                  %32 = arith.andi %29, %31 : i1
                  %33 = arith.cmpi slt, %arg8, %30 : index
                  %34 = arith.andi %32, %33 : i1
                  %35 = scf.if %34 -> (f32) {
                    %36 = memref.load %alloca_3[%arg7, %arg8] : memref<4x8xbf16>
                    %37 = arith.extf %36 : bf16 to f32
                    scf.yield %37 : f32
                  } else {
                    %cst_49 = arith.constant 0.000000e+00 : f32
                    scf.yield %cst_49 : f32
                  }
                  memref.store %35, %alloca_0[%arg7, %arg8] : memref<4x8xf32>
                }
              }
            }
            %c0_i32_25 = arith.constant 0 : i32
            %c0_26 = arith.constant 0 : index
            %c8_i32 = arith.constant 8 : i32
            %8 = arith.index_cast %c8_i32 : i32 to index
            %c1_27 = arith.constant 1 : index
            scf.for %arg7 = %c0_26 to %8 step %c1_27 {
              %14 = arith.sitofp %c0_i32_25 : i32 to f32
              memref.store %14, %alloca[%arg7] : memref<8xf32>
            }
            %c0_i32_28 = arith.constant 0 : i32
            %9 = arith.index_cast %c0_i32_28 : i32 to index
            %c4_i32_29 = arith.constant 4 : i32
            %10 = arith.index_cast %c4_i32_29 : i32 to index
            %c1_30 = arith.constant 1 : index
            %11 = arith.addi %9, %10 : index
            scf.for %arg7 = %9 to %11 step %c1_30 {
              %c0_i32_37 = arith.constant 0 : i32
              %14 = arith.index_cast %c0_i32_37 : i32 to index
              %c8_i32_38 = arith.constant 8 : i32
              %15 = arith.index_cast %c8_i32_38 : i32 to index
              %c1_39 = arith.constant 1 : index
              %16 = arith.addi %14, %15 : index
              scf.parallel (%arg8) = (%14) to (%16) step (%c1_39) {
                %17 = memref.load %alloca[%arg8] : memref<8xf32>
                %18 = memref.load %alloca_1[%arg7] : memref<4xf32>
                %19 = memref.load %alloca_0[%arg7, %arg8] : memref<4x8xf32>
                %20 = arith.mulf %18, %19 : f32
                %21 = arith.addf %17, %20 : f32
                memref.store %21, %alloca[%arg8] : memref<8xf32>
                scf.reduce 
              }
            }
            %c0_31 = arith.constant 0 : index
            %c8_i32_32 = arith.constant 8 : i32
            %12 = arith.index_cast %c8_i32_32 : i32 to index
            %c1_33 = arith.constant 1 : index
            scf.for %arg7 = %c0_31 to %12 step %c1_33 {
              %true = arith.constant true
              %c8_i32_37 = arith.constant 8 : i32
              %14 = arith.index_cast %c8_i32_37 : i32 to index
              %c0_38 = arith.constant 0 : index
              %15 = arith.cmpi sge, %arg7, %c0_38 : index
              %16 = arith.andi %true, %15 : i1
              %17 = arith.cmpi slt, %arg7, %14 : index
              %18 = arith.andi %16, %17 : i1
              scf.if %18 {
                %true_39 = arith.constant true
                %c8_i32_40 = arith.constant 8 : i32
                %19 = arith.index_cast %c8_i32_40 : i32 to index
                %c0_41 = arith.constant 0 : index
                %20 = arith.cmpi sge, %arg7, %c0_41 : index
                %21 = arith.andi %true_39, %20 : i1
                %22 = arith.cmpi slt, %arg7, %19 : index
                %23 = arith.andi %21, %22 : i1
                %24 = scf.if %23 -> (bf16) {
                  %25 = memref.load %alloca[%arg7] : memref<8xf32>
                  %26 = arith.truncf %25 : f32 to bf16
                  scf.yield %26 : bf16
                } else {
                  %cst_42 = arith.constant 0.000000e+00 : bf16
                  scf.yield %cst_42 : bf16
                }
                memref.store %24, %alloca_2[%arg7] : memref<8xbf16>
              }
            }
            %c0_34 = arith.constant 0 : index
            %c8_i32_35 = arith.constant 8 : i32
            %13 = arith.index_cast %c8_i32_35 : i32 to index
            %c1_36 = arith.constant 1 : index
            scf.for %arg7 = %c0_34 to %13 step %c1_36 {
              %true = arith.constant true
              %c0_37 = arith.constant 0 : index
              %14 = arith.cmpi sge, %arg3, %c0_37 : index
              %15 = arith.andi %true, %14 : i1
              %16 = arith.cmpi slt, %arg3, %dim : index
              %17 = arith.andi %15, %16 : i1
              %c8_i32_38 = arith.constant 8 : i32
              %18 = arith.index_cast %c8_i32_38 : i32 to index
              %c0_39 = arith.constant 0 : index
              %19 = arith.cmpi sge, %arg7, %c0_39 : index
              %20 = arith.andi %17, %19 : i1
              %21 = arith.cmpi slt, %arg7, %18 : index
              %22 = arith.andi %20, %21 : i1
              scf.if %22 {
                %true_40 = arith.constant true
                %c8_i32_41 = arith.constant 8 : i32
                %23 = arith.index_cast %c8_i32_41 : i32 to index
                %c0_42 = arith.constant 0 : index
                %24 = arith.cmpi sge, %arg7, %c0_42 : index
                %25 = arith.andi %true_40, %24 : i1
                %26 = arith.cmpi slt, %arg7, %23 : index
                %27 = arith.andi %25, %26 : i1
                %28 = scf.if %27 -> (bf16) {
                  %29 = memref.load %alloca_2[%arg7] : memref<8xbf16>
                  scf.yield %29 : bf16
                } else {
                  %cst_43 = arith.constant 0.000000e+00 : bf16
                  scf.yield %cst_43 : bf16
                }
                memref.store %28, %arg2[%arg3, %arg7] : memref<?x8xbf16>
              }
            }
          }
        }
      }
    }
    return
  }
}

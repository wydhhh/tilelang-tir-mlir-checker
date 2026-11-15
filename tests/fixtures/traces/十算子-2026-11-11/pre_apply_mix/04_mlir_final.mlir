module {
  func.func @_mhc_pre_apply_mix_fwd_kernel(%arg0: memref<?x4x8xbf16>, %arg1: memref<?x4xf32>, %arg2: memref<?x8xbf16>) {
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloca = memref.alloca() : memref<8xf32>
    %alloca_1 = memref.alloca() : memref<4x8xf32>
    %alloca_2 = memref.alloca() : memref<4xf32>
    %alloca_3 = memref.alloca() : memref<8xbf16>
    %alloca_4 = memref.alloca() : memref<4x8xbf16>
    %dim = memref.dim %arg0, %c0 : memref<?x4x8xbf16>
    scf.for %arg3 = %c0 to %dim step %c1 {
      scf.for %arg4 = %c0 to %c4 step %c1 {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          memref.store %cst_0, %alloca_4[%arg4, %arg5] : memref<4x8xbf16>
        }
      }
      scf.for %arg4 = %c0 to %c8 step %c1 {
        memref.store %cst_0, %alloca_3[%arg4] : memref<8xbf16>
      }
      scf.for %arg4 = %c0 to %c4 step %c1 {
        memref.store %cst, %alloca_2[%arg4] : memref<4xf32>
      }
      scf.for %arg4 = %c0 to %c4 step %c1 {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          memref.store %cst, %alloca_1[%arg4, %arg5] : memref<4x8xf32>
        }
      }
      scf.for %arg4 = %c0 to %c8 step %c1 {
        memref.store %cst, %alloca[%arg4] : memref<8xf32>
      }
      scf.for %arg4 = %c0 to %c4 step %c1 {
        %0 = arith.cmpi sge, %arg4, %c0 : index
        %1 = arith.cmpi slt, %arg4, %c4 : index
        %2 = arith.andi %0, %1 : i1
        scf.if %2 {
          %3 = arith.cmpi sge, %arg3, %c0 : index
          %4 = arith.cmpi slt, %arg3, %dim : index
          %5 = arith.andi %3, %4 : i1
          %6 = arith.andi %5, %0 : i1
          %7 = arith.andi %6, %1 : i1
          %8 = scf.if %7 -> (f32) {
            %9 = memref.load %arg1[%arg3, %arg4] : memref<?x4xf32>
            scf.yield %9 : f32
          } else {
            scf.yield %cst : f32
          }
          memref.store %8, %alloca_2[%arg4] : memref<4xf32>
        }
      }
      scf.for %arg4 = %c0 to %c4 step %c1 {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          %0 = arith.cmpi sge, %arg4, %c0 : index
          %1 = arith.cmpi slt, %arg4, %c4 : index
          %2 = arith.andi %0, %1 : i1
          %3 = arith.cmpi sge, %arg5, %c0 : index
          %4 = arith.andi %2, %3 : i1
          %5 = arith.cmpi slt, %arg5, %c8 : index
          %6 = arith.andi %4, %5 : i1
          scf.if %6 {
            %7 = arith.cmpi sge, %arg3, %c0 : index
            %8 = arith.cmpi slt, %arg3, %dim : index
            %9 = arith.andi %7, %8 : i1
            %10 = arith.andi %9, %0 : i1
            %11 = arith.andi %10, %1 : i1
            %12 = arith.andi %11, %3 : i1
            %13 = arith.andi %12, %5 : i1
            %14 = scf.if %13 -> (bf16) {
              %15 = memref.load %arg0[%arg3, %arg4, %arg5] : memref<?x4x8xbf16>
              scf.yield %15 : bf16
            } else {
              scf.yield %cst_0 : bf16
            }
            memref.store %14, %alloca_4[%arg4, %arg5] : memref<4x8xbf16>
          }
        }
      }
      scf.for %arg4 = %c0 to %c4 step %c1 {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          %0 = arith.cmpi sge, %arg4, %c0 : index
          %1 = arith.cmpi slt, %arg4, %c4 : index
          %2 = arith.andi %0, %1 : i1
          %3 = arith.cmpi sge, %arg5, %c0 : index
          %4 = arith.andi %2, %3 : i1
          %5 = arith.cmpi slt, %arg5, %c8 : index
          %6 = arith.andi %4, %5 : i1
          scf.if %6 {
            %7 = memref.load %alloca_4[%arg4, %arg5] : memref<4x8xbf16>
            %8 = arith.extf %7 : bf16 to f32
            memref.store %8, %alloca_1[%arg4, %arg5] : memref<4x8xf32>
          }
        }
      }
      scf.for %arg4 = %c0 to %c8 step %c1 {
        memref.store %cst, %alloca[%arg4] : memref<8xf32>
      }
      scf.for %arg4 = %c0 to %c4 step %c1 {
        scf.parallel (%arg5) = (%c0) to (%c8) step (%c1) {
          %0 = memref.load %alloca[%arg5] : memref<8xf32>
          %1 = memref.load %alloca_2[%arg4] : memref<4xf32>
          %2 = memref.load %alloca_1[%arg4, %arg5] : memref<4x8xf32>
          %3 = arith.mulf %1, %2 : f32
          %4 = arith.addf %0, %3 : f32
          memref.store %4, %alloca[%arg5] : memref<8xf32>
          scf.reduce 
        }
      }
      scf.for %arg4 = %c0 to %c8 step %c1 {
        %0 = arith.cmpi sge, %arg4, %c0 : index
        %1 = arith.cmpi slt, %arg4, %c8 : index
        %2 = arith.andi %0, %1 : i1
        scf.if %2 {
          %3 = memref.load %alloca[%arg4] : memref<8xf32>
          %4 = arith.truncf %3 : f32 to bf16
          memref.store %4, %alloca_3[%arg4] : memref<8xbf16>
        }
      }
      scf.for %arg4 = %c0 to %c8 step %c1 {
        %0 = arith.cmpi sge, %arg3, %c0 : index
        %1 = arith.cmpi slt, %arg3, %dim : index
        %2 = arith.andi %0, %1 : i1
        %3 = arith.cmpi sge, %arg4, %c0 : index
        %4 = arith.andi %2, %3 : i1
        %5 = arith.cmpi slt, %arg4, %c8 : index
        %6 = arith.andi %4, %5 : i1
        scf.if %6 {
          %7 = arith.andi %3, %5 : i1
          %8 = scf.if %7 -> (bf16) {
            %9 = memref.load %alloca_3[%arg4] : memref<8xbf16>
            scf.yield %9 : bf16
          } else {
            scf.yield %cst_0 : bf16
          }
          memref.store %8, %arg2[%arg3, %arg4] : memref<?x8xbf16>
        }
      }
    }
    return
  }
}


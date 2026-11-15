module {
  func.func @main(%arg0: memref<35xi8>, %arg1: memref<35xf32>, %arg2: memref<35xf32>, %arg3: memref<35xf32>) {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c35 = arith.constant 35 : index
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c4_i32 = arith.constant 4 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i8 = arith.constant 0 : i8
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloca = memref.alloca() : memref<32xf32>
    %alloca_0 = memref.alloca() : memref<32xf32>
    %alloca_1 = memref.alloca() : memref<32xi8>
    scf.for %arg4 = %c0 to %c2 step %c1 {
      scf.for %arg5 = %c0 to %c32 step %c1 {
        memref.store %c0_i8, %alloca_1[%arg5] : memref<32xi8>
      }
      scf.for %arg5 = %c0 to %c32 step %c1 {
        memref.store %cst, %alloca_0[%arg5] : memref<32xf32>
      }
      scf.for %arg5 = %c0 to %c32 step %c1 {
        memref.store %cst, %alloca[%arg5] : memref<32xf32>
      }
      scf.for %arg5 = %c0 to %c32 step %c1 {
        %0 = arith.index_cast %arg4 : index to i32
        %1 = arith.muli %0, %c32_i32 : i32
        %2 = arith.index_cast %1 : i32 to index
        %3 = arith.addi %arg5, %2 : index
        %4 = arith.cmpi sge, %arg5, %c0 : index
        %5 = arith.cmpi slt, %arg5, %c32 : index
        %6 = arith.andi %4, %5 : i1
        scf.if %6 {
          %7 = arith.cmpi sge, %3, %c0 : index
          %8 = arith.cmpi slt, %3, %c35 : index
          %9 = arith.andi %7, %8 : i1
          %10 = scf.if %9 -> (i8) {
            %11 = memref.load %arg0[%3] : memref<35xi8>
            scf.yield %11 : i8
          } else {
            scf.yield %c0_i8 : i8
          }
          memref.store %10, %alloca_1[%arg5] : memref<32xi8>
        }
      }
      scf.for %arg5 = %c0 to %c32 step %c1 {
        %0 = arith.index_cast %arg4 : index to i32
        %1 = arith.muli %0, %c32_i32 : i32
        %2 = arith.index_cast %1 : i32 to index
        %3 = arith.addi %arg5, %2 : index
        %4 = arith.cmpi sge, %arg5, %c0 : index
        %5 = arith.cmpi slt, %arg5, %c32 : index
        %6 = arith.andi %4, %5 : i1
        scf.if %6 {
          %7 = arith.cmpi sge, %3, %c0 : index
          %8 = arith.cmpi slt, %3, %c35 : index
          %9 = arith.andi %7, %8 : i1
          %10 = scf.if %9 -> (f32) {
            %11 = memref.load %arg1[%3] : memref<35xf32>
            scf.yield %11 : f32
          } else {
            scf.yield %cst : f32
          }
          memref.store %10, %alloca_0[%arg5] : memref<32xf32>
        }
      }
      scf.for %arg5 = %c0 to %c32 step %c1 {
        %0 = arith.index_cast %arg4 : index to i32
        %1 = arith.muli %0, %c32_i32 : i32
        %2 = arith.index_cast %1 : i32 to index
        %3 = arith.addi %arg5, %2 : index
        %4 = arith.cmpi sge, %arg5, %c0 : index
        %5 = arith.cmpi slt, %arg5, %c32 : index
        %6 = arith.andi %4, %5 : i1
        scf.if %6 {
          %7 = arith.cmpi sge, %3, %c0 : index
          %8 = arith.cmpi slt, %3, %c35 : index
          %9 = arith.andi %7, %8 : i1
          %10 = scf.if %9 -> (f32) {
            %11 = memref.load %arg2[%3] : memref<35xf32>
            scf.yield %11 : f32
          } else {
            scf.yield %cst : f32
          }
          memref.store %10, %alloca[%arg5] : memref<32xf32>
        }
      }
      scf.parallel (%arg5, %arg6) = (%c0, %c0) to (%c8, %c4) step (%c1, %c1) {
        %0 = arith.index_cast %arg5 : index to i32
        %1 = arith.muli %0, %c4_i32 : i32
        %2 = arith.index_cast %arg6 : index to i32
        %3 = arith.addi %1, %2 : i32
        %4 = arith.index_cast %3 : i32 to index
        %5 = memref.load %alloca_1[%4] : memref<32xi8>
        %6 = arith.cmpi ne, %5, %c0_i8 : i8
        %7 = scf.if %6 -> (f32) {
          %8 = memref.load %alloca_0[%4] : memref<32xf32>
          scf.yield %8 : f32
        } else {
          %8 = memref.load %alloca[%4] : memref<32xf32>
          scf.yield %8 : f32
        }
        memref.store %7, %alloca_0[%4] : memref<32xf32>
        scf.reduce 
      }
      scf.for %arg5 = %c0 to %c32 step %c1 {
        %0 = arith.index_cast %arg4 : index to i32
        %1 = arith.muli %0, %c32_i32 : i32
        %2 = arith.index_cast %1 : i32 to index
        %3 = arith.addi %arg5, %2 : index
        %4 = arith.cmpi sge, %3, %c0 : index
        %5 = arith.cmpi slt, %3, %c35 : index
        %6 = arith.andi %4, %5 : i1
        scf.if %6 {
          %7 = arith.cmpi sge, %arg5, %c0 : index
          %8 = arith.cmpi slt, %arg5, %c32 : index
          %9 = arith.andi %7, %8 : i1
          %10 = scf.if %9 -> (f32) {
            %11 = memref.load %alloca_0[%arg5] : memref<32xf32>
            scf.yield %11 : f32
          } else {
            scf.yield %cst : f32
          }
          memref.store %10, %arg3[%3] : memref<35xf32>
        }
      }
    }
    return
  }
}


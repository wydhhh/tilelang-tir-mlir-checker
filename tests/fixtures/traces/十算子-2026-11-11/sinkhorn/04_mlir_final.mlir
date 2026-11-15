module {
  func.func @mhc_sinkhorn_kernel(%arg0: memref<?x4x4xf32>, %arg1: memref<?x4x4xf32>) {
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 9.99999997E-7 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %alloca = memref.alloca() : memref<2x4xf32>
    %alloca_1 = memref.alloca() : memref<2x4xf32>
    %alloca_2 = memref.alloca() : memref<2x4xf32>
    %alloca_3 = memref.alloca() : memref<2x4x4xf32>
    %dim = memref.dim %arg0, %c0 : memref<?x4x4xf32>
    %0 = arith.index_cast %dim : index to i32
    %1 = arith.addi %0, %c1_i32 : i32
    %2 = arith.floordivsi %1, %c2_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    scf.for %arg2 = %c0 to %3 step %c1 {
      scf.for %arg3 = %c0 to %c2 step %c1 {
        scf.for %arg4 = %c0 to %c4 step %c1 {
          scf.for %arg5 = %c0 to %c4 step %c1 {
            memref.store %cst_0, %alloca_3[%arg3, %arg4, %arg5] : memref<2x4x4xf32>
          }
        }
      }
      scf.for %arg3 = %c0 to %c2 step %c1 {
        scf.for %arg4 = %c0 to %c4 step %c1 {
          memref.store %cst_0, %alloca_2[%arg3, %arg4] : memref<2x4xf32>
        }
      }
      scf.for %arg3 = %c0 to %c2 step %c1 {
        scf.for %arg4 = %c0 to %c4 step %c1 {
          memref.store %cst_0, %alloca_1[%arg3, %arg4] : memref<2x4xf32>
        }
      }
      scf.for %arg3 = %c0 to %c2 step %c1 {
        scf.for %arg4 = %c0 to %c4 step %c1 {
          memref.store %cst_0, %alloca[%arg3, %arg4] : memref<2x4xf32>
        }
      }
      scf.for %arg3 = %c0 to %c2 step %c1 {
        scf.for %arg4 = %c0 to %c4 step %c1 {
          scf.for %arg5 = %c0 to %c4 step %c1 {
            %212 = arith.index_cast %arg2 : index to i32
            %213 = arith.muli %212, %c2_i32 : i32
            %214 = arith.index_cast %213 : i32 to index
            %215 = arith.addi %arg3, %214 : index
            %216 = arith.cmpi sge, %arg3, %c0 : index
            %217 = arith.cmpi slt, %arg3, %c2 : index
            %218 = arith.andi %216, %217 : i1
            %219 = arith.cmpi sge, %arg4, %c0 : index
            %220 = arith.andi %218, %219 : i1
            %221 = arith.cmpi slt, %arg4, %c4 : index
            %222 = arith.andi %220, %221 : i1
            %223 = arith.cmpi sge, %arg5, %c0 : index
            %224 = arith.andi %222, %223 : i1
            %225 = arith.cmpi slt, %arg5, %c4 : index
            %226 = arith.andi %224, %225 : i1
            scf.if %226 {
              %227 = arith.cmpi sge, %215, %c0 : index
              %228 = arith.cmpi slt, %215, %dim : index
              %229 = arith.andi %227, %228 : i1
              %230 = arith.andi %229, %219 : i1
              %231 = arith.andi %230, %221 : i1
              %232 = arith.andi %231, %223 : i1
              %233 = arith.andi %232, %225 : i1
              %234 = scf.if %233 -> (f32) {
                %235 = memref.load %arg0[%215, %arg4, %arg5] : memref<?x4x4xf32>
                scf.yield %235 : f32
              } else {
                scf.yield %cst_0 : f32
              }
              memref.store %234, %alloca_3[%arg3, %arg4, %arg5] : memref<2x4x4xf32>
            }
          }
        }
      }
      %4 = memref.load %alloca_3[%c0, %c0, %c0] : memref<2x4x4xf32>
      %5 = memref.load %alloca_3[%c0, %c0, %c1] : memref<2x4x4xf32>
      %6 = arith.cmpf ogt, %4, %5 : f32
      %7 = arith.select %6, %4, %5 : f32
      %8 = memref.load %alloca_3[%c0, %c0, %c2] : memref<2x4x4xf32>
      %9 = arith.cmpf ogt, %7, %8 : f32
      %10 = arith.select %9, %7, %8 : f32
      %11 = memref.load %alloca_3[%c0, %c0, %c3] : memref<2x4x4xf32>
      %12 = arith.cmpf ogt, %10, %11 : f32
      %13 = arith.select %12, %10, %11 : f32
      memref.store %13, %alloca[%c0, %c0] : memref<2x4xf32>
      %14 = memref.load %alloca_3[%c0, %c1, %c0] : memref<2x4x4xf32>
      %15 = memref.load %alloca_3[%c0, %c1, %c1] : memref<2x4x4xf32>
      %16 = arith.cmpf ogt, %14, %15 : f32
      %17 = arith.select %16, %14, %15 : f32
      %18 = memref.load %alloca_3[%c0, %c1, %c2] : memref<2x4x4xf32>
      %19 = arith.cmpf ogt, %17, %18 : f32
      %20 = arith.select %19, %17, %18 : f32
      %21 = memref.load %alloca_3[%c0, %c1, %c3] : memref<2x4x4xf32>
      %22 = arith.cmpf ogt, %20, %21 : f32
      %23 = arith.select %22, %20, %21 : f32
      memref.store %23, %alloca[%c0, %c1] : memref<2x4xf32>
      %24 = memref.load %alloca_3[%c0, %c2, %c0] : memref<2x4x4xf32>
      %25 = memref.load %alloca_3[%c0, %c2, %c1] : memref<2x4x4xf32>
      %26 = arith.cmpf ogt, %24, %25 : f32
      %27 = arith.select %26, %24, %25 : f32
      %28 = memref.load %alloca_3[%c0, %c2, %c2] : memref<2x4x4xf32>
      %29 = arith.cmpf ogt, %27, %28 : f32
      %30 = arith.select %29, %27, %28 : f32
      %31 = memref.load %alloca_3[%c0, %c2, %c3] : memref<2x4x4xf32>
      %32 = arith.cmpf ogt, %30, %31 : f32
      %33 = arith.select %32, %30, %31 : f32
      memref.store %33, %alloca[%c0, %c2] : memref<2x4xf32>
      %34 = memref.load %alloca_3[%c0, %c3, %c0] : memref<2x4x4xf32>
      %35 = memref.load %alloca_3[%c0, %c3, %c1] : memref<2x4x4xf32>
      %36 = arith.cmpf ogt, %34, %35 : f32
      %37 = arith.select %36, %34, %35 : f32
      %38 = memref.load %alloca_3[%c0, %c3, %c2] : memref<2x4x4xf32>
      %39 = arith.cmpf ogt, %37, %38 : f32
      %40 = arith.select %39, %37, %38 : f32
      %41 = memref.load %alloca_3[%c0, %c3, %c3] : memref<2x4x4xf32>
      %42 = arith.cmpf ogt, %40, %41 : f32
      %43 = arith.select %42, %40, %41 : f32
      memref.store %43, %alloca[%c0, %c3] : memref<2x4xf32>
      %44 = memref.load %alloca_3[%c1, %c0, %c0] : memref<2x4x4xf32>
      %45 = memref.load %alloca_3[%c1, %c0, %c1] : memref<2x4x4xf32>
      %46 = arith.cmpf ogt, %44, %45 : f32
      %47 = arith.select %46, %44, %45 : f32
      %48 = memref.load %alloca_3[%c1, %c0, %c2] : memref<2x4x4xf32>
      %49 = arith.cmpf ogt, %47, %48 : f32
      %50 = arith.select %49, %47, %48 : f32
      %51 = memref.load %alloca_3[%c1, %c0, %c3] : memref<2x4x4xf32>
      %52 = arith.cmpf ogt, %50, %51 : f32
      %53 = arith.select %52, %50, %51 : f32
      memref.store %53, %alloca[%c1, %c0] : memref<2x4xf32>
      %54 = memref.load %alloca_3[%c1, %c1, %c0] : memref<2x4x4xf32>
      %55 = memref.load %alloca_3[%c1, %c1, %c1] : memref<2x4x4xf32>
      %56 = arith.cmpf ogt, %54, %55 : f32
      %57 = arith.select %56, %54, %55 : f32
      %58 = memref.load %alloca_3[%c1, %c1, %c2] : memref<2x4x4xf32>
      %59 = arith.cmpf ogt, %57, %58 : f32
      %60 = arith.select %59, %57, %58 : f32
      %61 = memref.load %alloca_3[%c1, %c1, %c3] : memref<2x4x4xf32>
      %62 = arith.cmpf ogt, %60, %61 : f32
      %63 = arith.select %62, %60, %61 : f32
      memref.store %63, %alloca[%c1, %c1] : memref<2x4xf32>
      %64 = memref.load %alloca_3[%c1, %c2, %c0] : memref<2x4x4xf32>
      %65 = memref.load %alloca_3[%c1, %c2, %c1] : memref<2x4x4xf32>
      %66 = arith.cmpf ogt, %64, %65 : f32
      %67 = arith.select %66, %64, %65 : f32
      %68 = memref.load %alloca_3[%c1, %c2, %c2] : memref<2x4x4xf32>
      %69 = arith.cmpf ogt, %67, %68 : f32
      %70 = arith.select %69, %67, %68 : f32
      %71 = memref.load %alloca_3[%c1, %c2, %c3] : memref<2x4x4xf32>
      %72 = arith.cmpf ogt, %70, %71 : f32
      %73 = arith.select %72, %70, %71 : f32
      memref.store %73, %alloca[%c1, %c2] : memref<2x4xf32>
      %74 = memref.load %alloca_3[%c1, %c3, %c0] : memref<2x4x4xf32>
      %75 = memref.load %alloca_3[%c1, %c3, %c1] : memref<2x4x4xf32>
      %76 = arith.cmpf ogt, %74, %75 : f32
      %77 = arith.select %76, %74, %75 : f32
      %78 = memref.load %alloca_3[%c1, %c3, %c2] : memref<2x4x4xf32>
      %79 = arith.cmpf ogt, %77, %78 : f32
      %80 = arith.select %79, %77, %78 : f32
      %81 = memref.load %alloca_3[%c1, %c3, %c3] : memref<2x4x4xf32>
      %82 = arith.cmpf ogt, %80, %81 : f32
      %83 = arith.select %82, %80, %81 : f32
      memref.store %83, %alloca[%c1, %c3] : memref<2x4xf32>
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c4) step (%c1, %c1, %c1) {
        %212 = memref.load %alloca_3[%arg3, %arg4, %arg5] : memref<2x4x4xf32>
        %213 = memref.load %alloca[%arg3, %arg4] : memref<2x4xf32>
        %214 = arith.subf %212, %213 : f32
        %215 = math.exp %214 : f32
        memref.store %215, %alloca_3[%arg3, %arg4, %arg5] : memref<2x4x4xf32>
        scf.reduce 
      }
      %84 = memref.load %alloca_3[%c0, %c0, %c0] : memref<2x4x4xf32>
      %85 = arith.addf %84, %cst_0 : f32
      %86 = memref.load %alloca_3[%c0, %c0, %c1] : memref<2x4x4xf32>
      %87 = arith.addf %85, %86 : f32
      %88 = memref.load %alloca_3[%c0, %c0, %c2] : memref<2x4x4xf32>
      %89 = arith.addf %87, %88 : f32
      %90 = memref.load %alloca_3[%c0, %c0, %c3] : memref<2x4x4xf32>
      %91 = arith.addf %89, %90 : f32
      memref.store %91, %alloca_2[%c0, %c0] : memref<2x4xf32>
      %92 = memref.load %alloca_3[%c0, %c1, %c0] : memref<2x4x4xf32>
      %93 = arith.addf %92, %cst_0 : f32
      %94 = memref.load %alloca_3[%c0, %c1, %c1] : memref<2x4x4xf32>
      %95 = arith.addf %93, %94 : f32
      %96 = memref.load %alloca_3[%c0, %c1, %c2] : memref<2x4x4xf32>
      %97 = arith.addf %95, %96 : f32
      %98 = memref.load %alloca_3[%c0, %c1, %c3] : memref<2x4x4xf32>
      %99 = arith.addf %97, %98 : f32
      memref.store %99, %alloca_2[%c0, %c1] : memref<2x4xf32>
      %100 = memref.load %alloca_3[%c0, %c2, %c0] : memref<2x4x4xf32>
      %101 = arith.addf %100, %cst_0 : f32
      %102 = memref.load %alloca_3[%c0, %c2, %c1] : memref<2x4x4xf32>
      %103 = arith.addf %101, %102 : f32
      %104 = memref.load %alloca_3[%c0, %c2, %c2] : memref<2x4x4xf32>
      %105 = arith.addf %103, %104 : f32
      %106 = memref.load %alloca_3[%c0, %c2, %c3] : memref<2x4x4xf32>
      %107 = arith.addf %105, %106 : f32
      memref.store %107, %alloca_2[%c0, %c2] : memref<2x4xf32>
      %108 = memref.load %alloca_3[%c0, %c3, %c0] : memref<2x4x4xf32>
      %109 = arith.addf %108, %cst_0 : f32
      %110 = memref.load %alloca_3[%c0, %c3, %c1] : memref<2x4x4xf32>
      %111 = arith.addf %109, %110 : f32
      %112 = memref.load %alloca_3[%c0, %c3, %c2] : memref<2x4x4xf32>
      %113 = arith.addf %111, %112 : f32
      %114 = memref.load %alloca_3[%c0, %c3, %c3] : memref<2x4x4xf32>
      %115 = arith.addf %113, %114 : f32
      memref.store %115, %alloca_2[%c0, %c3] : memref<2x4xf32>
      %116 = memref.load %alloca_3[%c1, %c0, %c0] : memref<2x4x4xf32>
      %117 = arith.addf %116, %cst_0 : f32
      %118 = memref.load %alloca_3[%c1, %c0, %c1] : memref<2x4x4xf32>
      %119 = arith.addf %117, %118 : f32
      %120 = memref.load %alloca_3[%c1, %c0, %c2] : memref<2x4x4xf32>
      %121 = arith.addf %119, %120 : f32
      %122 = memref.load %alloca_3[%c1, %c0, %c3] : memref<2x4x4xf32>
      %123 = arith.addf %121, %122 : f32
      memref.store %123, %alloca_2[%c1, %c0] : memref<2x4xf32>
      %124 = memref.load %alloca_3[%c1, %c1, %c0] : memref<2x4x4xf32>
      %125 = arith.addf %124, %cst_0 : f32
      %126 = memref.load %alloca_3[%c1, %c1, %c1] : memref<2x4x4xf32>
      %127 = arith.addf %125, %126 : f32
      %128 = memref.load %alloca_3[%c1, %c1, %c2] : memref<2x4x4xf32>
      %129 = arith.addf %127, %128 : f32
      %130 = memref.load %alloca_3[%c1, %c1, %c3] : memref<2x4x4xf32>
      %131 = arith.addf %129, %130 : f32
      memref.store %131, %alloca_2[%c1, %c1] : memref<2x4xf32>
      %132 = memref.load %alloca_3[%c1, %c2, %c0] : memref<2x4x4xf32>
      %133 = arith.addf %132, %cst_0 : f32
      %134 = memref.load %alloca_3[%c1, %c2, %c1] : memref<2x4x4xf32>
      %135 = arith.addf %133, %134 : f32
      %136 = memref.load %alloca_3[%c1, %c2, %c2] : memref<2x4x4xf32>
      %137 = arith.addf %135, %136 : f32
      %138 = memref.load %alloca_3[%c1, %c2, %c3] : memref<2x4x4xf32>
      %139 = arith.addf %137, %138 : f32
      memref.store %139, %alloca_2[%c1, %c2] : memref<2x4xf32>
      %140 = memref.load %alloca_3[%c1, %c3, %c0] : memref<2x4x4xf32>
      %141 = arith.addf %140, %cst_0 : f32
      %142 = memref.load %alloca_3[%c1, %c3, %c1] : memref<2x4x4xf32>
      %143 = arith.addf %141, %142 : f32
      %144 = memref.load %alloca_3[%c1, %c3, %c2] : memref<2x4x4xf32>
      %145 = arith.addf %143, %144 : f32
      %146 = memref.load %alloca_3[%c1, %c3, %c3] : memref<2x4x4xf32>
      %147 = arith.addf %145, %146 : f32
      memref.store %147, %alloca_2[%c1, %c3] : memref<2x4xf32>
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c4) step (%c1, %c1, %c1) {
        %212 = memref.load %alloca_3[%arg3, %arg4, %arg5] : memref<2x4x4xf32>
        %213 = memref.load %alloca_2[%arg3, %arg4] : memref<2x4xf32>
        %214 = arith.divf %212, %213 : f32
        %215 = arith.addf %214, %cst : f32
        memref.store %215, %alloca_3[%arg3, %arg4, %arg5] : memref<2x4x4xf32>
        scf.reduce 
      }
      %148 = memref.load %alloca_3[%c0, %c0, %c0] : memref<2x4x4xf32>
      %149 = arith.addf %148, %cst_0 : f32
      %150 = memref.load %alloca_3[%c0, %c1, %c0] : memref<2x4x4xf32>
      %151 = arith.addf %149, %150 : f32
      %152 = memref.load %alloca_3[%c0, %c2, %c0] : memref<2x4x4xf32>
      %153 = arith.addf %151, %152 : f32
      %154 = memref.load %alloca_3[%c0, %c3, %c0] : memref<2x4x4xf32>
      %155 = arith.addf %153, %154 : f32
      memref.store %155, %alloca_1[%c0, %c0] : memref<2x4xf32>
      %156 = memref.load %alloca_3[%c0, %c0, %c1] : memref<2x4x4xf32>
      %157 = arith.addf %156, %cst_0 : f32
      %158 = memref.load %alloca_3[%c0, %c1, %c1] : memref<2x4x4xf32>
      %159 = arith.addf %157, %158 : f32
      %160 = memref.load %alloca_3[%c0, %c2, %c1] : memref<2x4x4xf32>
      %161 = arith.addf %159, %160 : f32
      %162 = memref.load %alloca_3[%c0, %c3, %c1] : memref<2x4x4xf32>
      %163 = arith.addf %161, %162 : f32
      memref.store %163, %alloca_1[%c0, %c1] : memref<2x4xf32>
      %164 = memref.load %alloca_3[%c0, %c0, %c2] : memref<2x4x4xf32>
      %165 = arith.addf %164, %cst_0 : f32
      %166 = memref.load %alloca_3[%c0, %c1, %c2] : memref<2x4x4xf32>
      %167 = arith.addf %165, %166 : f32
      %168 = memref.load %alloca_3[%c0, %c2, %c2] : memref<2x4x4xf32>
      %169 = arith.addf %167, %168 : f32
      %170 = memref.load %alloca_3[%c0, %c3, %c2] : memref<2x4x4xf32>
      %171 = arith.addf %169, %170 : f32
      memref.store %171, %alloca_1[%c0, %c2] : memref<2x4xf32>
      %172 = memref.load %alloca_3[%c0, %c0, %c3] : memref<2x4x4xf32>
      %173 = arith.addf %172, %cst_0 : f32
      %174 = memref.load %alloca_3[%c0, %c1, %c3] : memref<2x4x4xf32>
      %175 = arith.addf %173, %174 : f32
      %176 = memref.load %alloca_3[%c0, %c2, %c3] : memref<2x4x4xf32>
      %177 = arith.addf %175, %176 : f32
      %178 = memref.load %alloca_3[%c0, %c3, %c3] : memref<2x4x4xf32>
      %179 = arith.addf %177, %178 : f32
      memref.store %179, %alloca_1[%c0, %c3] : memref<2x4xf32>
      %180 = memref.load %alloca_3[%c1, %c0, %c0] : memref<2x4x4xf32>
      %181 = arith.addf %180, %cst_0 : f32
      %182 = memref.load %alloca_3[%c1, %c1, %c0] : memref<2x4x4xf32>
      %183 = arith.addf %181, %182 : f32
      %184 = memref.load %alloca_3[%c1, %c2, %c0] : memref<2x4x4xf32>
      %185 = arith.addf %183, %184 : f32
      %186 = memref.load %alloca_3[%c1, %c3, %c0] : memref<2x4x4xf32>
      %187 = arith.addf %185, %186 : f32
      memref.store %187, %alloca_1[%c1, %c0] : memref<2x4xf32>
      %188 = memref.load %alloca_3[%c1, %c0, %c1] : memref<2x4x4xf32>
      %189 = arith.addf %188, %cst_0 : f32
      %190 = memref.load %alloca_3[%c1, %c1, %c1] : memref<2x4x4xf32>
      %191 = arith.addf %189, %190 : f32
      %192 = memref.load %alloca_3[%c1, %c2, %c1] : memref<2x4x4xf32>
      %193 = arith.addf %191, %192 : f32
      %194 = memref.load %alloca_3[%c1, %c3, %c1] : memref<2x4x4xf32>
      %195 = arith.addf %193, %194 : f32
      memref.store %195, %alloca_1[%c1, %c1] : memref<2x4xf32>
      %196 = memref.load %alloca_3[%c1, %c0, %c2] : memref<2x4x4xf32>
      %197 = arith.addf %196, %cst_0 : f32
      %198 = memref.load %alloca_3[%c1, %c1, %c2] : memref<2x4x4xf32>
      %199 = arith.addf %197, %198 : f32
      %200 = memref.load %alloca_3[%c1, %c2, %c2] : memref<2x4x4xf32>
      %201 = arith.addf %199, %200 : f32
      %202 = memref.load %alloca_3[%c1, %c3, %c2] : memref<2x4x4xf32>
      %203 = arith.addf %201, %202 : f32
      memref.store %203, %alloca_1[%c1, %c2] : memref<2x4xf32>
      %204 = memref.load %alloca_3[%c1, %c0, %c3] : memref<2x4x4xf32>
      %205 = arith.addf %204, %cst_0 : f32
      %206 = memref.load %alloca_3[%c1, %c1, %c3] : memref<2x4x4xf32>
      %207 = arith.addf %205, %206 : f32
      %208 = memref.load %alloca_3[%c1, %c2, %c3] : memref<2x4x4xf32>
      %209 = arith.addf %207, %208 : f32
      %210 = memref.load %alloca_3[%c1, %c3, %c3] : memref<2x4x4xf32>
      %211 = arith.addf %209, %210 : f32
      memref.store %211, %alloca_1[%c1, %c3] : memref<2x4xf32>
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c4) step (%c1, %c1, %c1) {
        %212 = memref.load %alloca_3[%arg3, %arg4, %arg5] : memref<2x4x4xf32>
        %213 = memref.load %alloca_1[%arg3, %arg5] : memref<2x4xf32>
        %214 = arith.addf %213, %cst : f32
        %215 = arith.divf %212, %214 : f32
        memref.store %215, %alloca_3[%arg3, %arg4, %arg5] : memref<2x4x4xf32>
        scf.reduce 
      }
      scf.for %arg3 = %c0 to %c2 step %c1 {
        %212 = memref.load %alloca_3[%c0, %c0, %c0] : memref<2x4x4xf32>
        %213 = arith.addf %212, %cst_0 : f32
        %214 = memref.load %alloca_3[%c0, %c0, %c1] : memref<2x4x4xf32>
        %215 = arith.addf %213, %214 : f32
        %216 = memref.load %alloca_3[%c0, %c0, %c2] : memref<2x4x4xf32>
        %217 = arith.addf %215, %216 : f32
        %218 = memref.load %alloca_3[%c0, %c0, %c3] : memref<2x4x4xf32>
        %219 = arith.addf %217, %218 : f32
        memref.store %219, %alloca_2[%c0, %c0] : memref<2x4xf32>
        %220 = memref.load %alloca_3[%c0, %c1, %c0] : memref<2x4x4xf32>
        %221 = arith.addf %220, %cst_0 : f32
        %222 = memref.load %alloca_3[%c0, %c1, %c1] : memref<2x4x4xf32>
        %223 = arith.addf %221, %222 : f32
        %224 = memref.load %alloca_3[%c0, %c1, %c2] : memref<2x4x4xf32>
        %225 = arith.addf %223, %224 : f32
        %226 = memref.load %alloca_3[%c0, %c1, %c3] : memref<2x4x4xf32>
        %227 = arith.addf %225, %226 : f32
        memref.store %227, %alloca_2[%c0, %c1] : memref<2x4xf32>
        %228 = memref.load %alloca_3[%c0, %c2, %c0] : memref<2x4x4xf32>
        %229 = arith.addf %228, %cst_0 : f32
        %230 = memref.load %alloca_3[%c0, %c2, %c1] : memref<2x4x4xf32>
        %231 = arith.addf %229, %230 : f32
        %232 = memref.load %alloca_3[%c0, %c2, %c2] : memref<2x4x4xf32>
        %233 = arith.addf %231, %232 : f32
        %234 = memref.load %alloca_3[%c0, %c2, %c3] : memref<2x4x4xf32>
        %235 = arith.addf %233, %234 : f32
        memref.store %235, %alloca_2[%c0, %c2] : memref<2x4xf32>
        %236 = memref.load %alloca_3[%c0, %c3, %c0] : memref<2x4x4xf32>
        %237 = arith.addf %236, %cst_0 : f32
        %238 = memref.load %alloca_3[%c0, %c3, %c1] : memref<2x4x4xf32>
        %239 = arith.addf %237, %238 : f32
        %240 = memref.load %alloca_3[%c0, %c3, %c2] : memref<2x4x4xf32>
        %241 = arith.addf %239, %240 : f32
        %242 = memref.load %alloca_3[%c0, %c3, %c3] : memref<2x4x4xf32>
        %243 = arith.addf %241, %242 : f32
        memref.store %243, %alloca_2[%c0, %c3] : memref<2x4xf32>
        %244 = memref.load %alloca_3[%c1, %c0, %c0] : memref<2x4x4xf32>
        %245 = arith.addf %244, %cst_0 : f32
        %246 = memref.load %alloca_3[%c1, %c0, %c1] : memref<2x4x4xf32>
        %247 = arith.addf %245, %246 : f32
        %248 = memref.load %alloca_3[%c1, %c0, %c2] : memref<2x4x4xf32>
        %249 = arith.addf %247, %248 : f32
        %250 = memref.load %alloca_3[%c1, %c0, %c3] : memref<2x4x4xf32>
        %251 = arith.addf %249, %250 : f32
        memref.store %251, %alloca_2[%c1, %c0] : memref<2x4xf32>
        %252 = memref.load %alloca_3[%c1, %c1, %c0] : memref<2x4x4xf32>
        %253 = arith.addf %252, %cst_0 : f32
        %254 = memref.load %alloca_3[%c1, %c1, %c1] : memref<2x4x4xf32>
        %255 = arith.addf %253, %254 : f32
        %256 = memref.load %alloca_3[%c1, %c1, %c2] : memref<2x4x4xf32>
        %257 = arith.addf %255, %256 : f32
        %258 = memref.load %alloca_3[%c1, %c1, %c3] : memref<2x4x4xf32>
        %259 = arith.addf %257, %258 : f32
        memref.store %259, %alloca_2[%c1, %c1] : memref<2x4xf32>
        %260 = memref.load %alloca_3[%c1, %c2, %c0] : memref<2x4x4xf32>
        %261 = arith.addf %260, %cst_0 : f32
        %262 = memref.load %alloca_3[%c1, %c2, %c1] : memref<2x4x4xf32>
        %263 = arith.addf %261, %262 : f32
        %264 = memref.load %alloca_3[%c1, %c2, %c2] : memref<2x4x4xf32>
        %265 = arith.addf %263, %264 : f32
        %266 = memref.load %alloca_3[%c1, %c2, %c3] : memref<2x4x4xf32>
        %267 = arith.addf %265, %266 : f32
        memref.store %267, %alloca_2[%c1, %c2] : memref<2x4xf32>
        %268 = memref.load %alloca_3[%c1, %c3, %c0] : memref<2x4x4xf32>
        %269 = arith.addf %268, %cst_0 : f32
        %270 = memref.load %alloca_3[%c1, %c3, %c1] : memref<2x4x4xf32>
        %271 = arith.addf %269, %270 : f32
        %272 = memref.load %alloca_3[%c1, %c3, %c2] : memref<2x4x4xf32>
        %273 = arith.addf %271, %272 : f32
        %274 = memref.load %alloca_3[%c1, %c3, %c3] : memref<2x4x4xf32>
        %275 = arith.addf %273, %274 : f32
        memref.store %275, %alloca_2[%c1, %c3] : memref<2x4xf32>
        scf.parallel (%arg4, %arg5, %arg6) = (%c0, %c0, %c0) to (%c2, %c4, %c4) step (%c1, %c1, %c1) {
          %340 = memref.load %alloca_3[%arg4, %arg5, %arg6] : memref<2x4x4xf32>
          %341 = memref.load %alloca_2[%arg4, %arg5] : memref<2x4xf32>
          %342 = arith.addf %341, %cst : f32
          %343 = arith.divf %340, %342 : f32
          memref.store %343, %alloca_3[%arg4, %arg5, %arg6] : memref<2x4x4xf32>
          scf.reduce 
        }
        %276 = memref.load %alloca_3[%c0, %c0, %c0] : memref<2x4x4xf32>
        %277 = arith.addf %276, %cst_0 : f32
        %278 = memref.load %alloca_3[%c0, %c1, %c0] : memref<2x4x4xf32>
        %279 = arith.addf %277, %278 : f32
        %280 = memref.load %alloca_3[%c0, %c2, %c0] : memref<2x4x4xf32>
        %281 = arith.addf %279, %280 : f32
        %282 = memref.load %alloca_3[%c0, %c3, %c0] : memref<2x4x4xf32>
        %283 = arith.addf %281, %282 : f32
        memref.store %283, %alloca_1[%c0, %c0] : memref<2x4xf32>
        %284 = memref.load %alloca_3[%c0, %c0, %c1] : memref<2x4x4xf32>
        %285 = arith.addf %284, %cst_0 : f32
        %286 = memref.load %alloca_3[%c0, %c1, %c1] : memref<2x4x4xf32>
        %287 = arith.addf %285, %286 : f32
        %288 = memref.load %alloca_3[%c0, %c2, %c1] : memref<2x4x4xf32>
        %289 = arith.addf %287, %288 : f32
        %290 = memref.load %alloca_3[%c0, %c3, %c1] : memref<2x4x4xf32>
        %291 = arith.addf %289, %290 : f32
        memref.store %291, %alloca_1[%c0, %c1] : memref<2x4xf32>
        %292 = memref.load %alloca_3[%c0, %c0, %c2] : memref<2x4x4xf32>
        %293 = arith.addf %292, %cst_0 : f32
        %294 = memref.load %alloca_3[%c0, %c1, %c2] : memref<2x4x4xf32>
        %295 = arith.addf %293, %294 : f32
        %296 = memref.load %alloca_3[%c0, %c2, %c2] : memref<2x4x4xf32>
        %297 = arith.addf %295, %296 : f32
        %298 = memref.load %alloca_3[%c0, %c3, %c2] : memref<2x4x4xf32>
        %299 = arith.addf %297, %298 : f32
        memref.store %299, %alloca_1[%c0, %c2] : memref<2x4xf32>
        %300 = memref.load %alloca_3[%c0, %c0, %c3] : memref<2x4x4xf32>
        %301 = arith.addf %300, %cst_0 : f32
        %302 = memref.load %alloca_3[%c0, %c1, %c3] : memref<2x4x4xf32>
        %303 = arith.addf %301, %302 : f32
        %304 = memref.load %alloca_3[%c0, %c2, %c3] : memref<2x4x4xf32>
        %305 = arith.addf %303, %304 : f32
        %306 = memref.load %alloca_3[%c0, %c3, %c3] : memref<2x4x4xf32>
        %307 = arith.addf %305, %306 : f32
        memref.store %307, %alloca_1[%c0, %c3] : memref<2x4xf32>
        %308 = memref.load %alloca_3[%c1, %c0, %c0] : memref<2x4x4xf32>
        %309 = arith.addf %308, %cst_0 : f32
        %310 = memref.load %alloca_3[%c1, %c1, %c0] : memref<2x4x4xf32>
        %311 = arith.addf %309, %310 : f32
        %312 = memref.load %alloca_3[%c1, %c2, %c0] : memref<2x4x4xf32>
        %313 = arith.addf %311, %312 : f32
        %314 = memref.load %alloca_3[%c1, %c3, %c0] : memref<2x4x4xf32>
        %315 = arith.addf %313, %314 : f32
        memref.store %315, %alloca_1[%c1, %c0] : memref<2x4xf32>
        %316 = memref.load %alloca_3[%c1, %c0, %c1] : memref<2x4x4xf32>
        %317 = arith.addf %316, %cst_0 : f32
        %318 = memref.load %alloca_3[%c1, %c1, %c1] : memref<2x4x4xf32>
        %319 = arith.addf %317, %318 : f32
        %320 = memref.load %alloca_3[%c1, %c2, %c1] : memref<2x4x4xf32>
        %321 = arith.addf %319, %320 : f32
        %322 = memref.load %alloca_3[%c1, %c3, %c1] : memref<2x4x4xf32>
        %323 = arith.addf %321, %322 : f32
        memref.store %323, %alloca_1[%c1, %c1] : memref<2x4xf32>
        %324 = memref.load %alloca_3[%c1, %c0, %c2] : memref<2x4x4xf32>
        %325 = arith.addf %324, %cst_0 : f32
        %326 = memref.load %alloca_3[%c1, %c1, %c2] : memref<2x4x4xf32>
        %327 = arith.addf %325, %326 : f32
        %328 = memref.load %alloca_3[%c1, %c2, %c2] : memref<2x4x4xf32>
        %329 = arith.addf %327, %328 : f32
        %330 = memref.load %alloca_3[%c1, %c3, %c2] : memref<2x4x4xf32>
        %331 = arith.addf %329, %330 : f32
        memref.store %331, %alloca_1[%c1, %c2] : memref<2x4xf32>
        %332 = memref.load %alloca_3[%c1, %c0, %c3] : memref<2x4x4xf32>
        %333 = arith.addf %332, %cst_0 : f32
        %334 = memref.load %alloca_3[%c1, %c1, %c3] : memref<2x4x4xf32>
        %335 = arith.addf %333, %334 : f32
        %336 = memref.load %alloca_3[%c1, %c2, %c3] : memref<2x4x4xf32>
        %337 = arith.addf %335, %336 : f32
        %338 = memref.load %alloca_3[%c1, %c3, %c3] : memref<2x4x4xf32>
        %339 = arith.addf %337, %338 : f32
        memref.store %339, %alloca_1[%c1, %c3] : memref<2x4xf32>
        scf.parallel (%arg4, %arg5, %arg6) = (%c0, %c0, %c0) to (%c2, %c4, %c4) step (%c1, %c1, %c1) {
          %340 = memref.load %alloca_3[%arg4, %arg5, %arg6] : memref<2x4x4xf32>
          %341 = memref.load %alloca_1[%arg4, %arg6] : memref<2x4xf32>
          %342 = arith.addf %341, %cst : f32
          %343 = arith.divf %340, %342 : f32
          memref.store %343, %alloca_3[%arg4, %arg5, %arg6] : memref<2x4x4xf32>
          scf.reduce 
        }
      }
      scf.for %arg3 = %c0 to %c2 step %c1 {
        scf.for %arg4 = %c0 to %c4 step %c1 {
          scf.for %arg5 = %c0 to %c4 step %c1 {
            %212 = arith.index_cast %arg2 : index to i32
            %213 = arith.muli %212, %c2_i32 : i32
            %214 = arith.index_cast %213 : i32 to index
            %215 = arith.addi %arg3, %214 : index
            %216 = arith.cmpi sge, %215, %c0 : index
            %217 = arith.cmpi slt, %215, %dim : index
            %218 = arith.andi %216, %217 : i1
            %219 = arith.cmpi sge, %arg4, %c0 : index
            %220 = arith.andi %218, %219 : i1
            %221 = arith.cmpi slt, %arg4, %c4 : index
            %222 = arith.andi %220, %221 : i1
            %223 = arith.cmpi sge, %arg5, %c0 : index
            %224 = arith.andi %222, %223 : i1
            %225 = arith.cmpi slt, %arg5, %c4 : index
            %226 = arith.andi %224, %225 : i1
            scf.if %226 {
              %227 = arith.cmpi sge, %arg3, %c0 : index
              %228 = arith.cmpi slt, %arg3, %c2 : index
              %229 = arith.andi %227, %228 : i1
              %230 = arith.andi %229, %219 : i1
              %231 = arith.andi %230, %221 : i1
              %232 = arith.andi %231, %223 : i1
              %233 = arith.andi %232, %225 : i1
              %234 = scf.if %233 -> (f32) {
                %235 = memref.load %alloca_3[%arg3, %arg4, %arg5] : memref<2x4x4xf32>
                scf.yield %235 : f32
              } else {
                scf.yield %cst_0 : f32
              }
              memref.store %234, %arg1[%215, %arg4, %arg5] : memref<?x4x4xf32>
            }
          }
        }
      }
    }
    return
  }
}


module {
  func.func @mhc_sinkhorn_kernel(%arg0: memref<?x4x4xf32>, %arg1: memref<?x4x4xf32>) {
    %alloca = memref.alloca() : memref<2x4xf32>
    %alloca_0 = memref.alloca() : memref<2x4xf32>
    %alloca_1 = memref.alloca() : memref<2x4xf32>
    %alloca_2 = memref.alloca() : memref<2x4x4xf32>
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?x4x4xf32>
    %c0_3 = arith.constant 0 : index
    %0 = arith.index_cast %dim : index to i32
    %c1_i32 = arith.constant 1 : i32
    %1 = arith.addi %0, %c1_i32 : i32
    %c2_i32 = arith.constant 2 : i32
    %2 = arith.floordivsi %1, %c2_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %c1 = arith.constant 1 : index
    scf.for %arg2 = %c0_3 to %3 step %c1 {
      %c0_i32 = arith.constant 0 : i32
      %c0_4 = arith.constant 0 : index
      %c1_i32_5 = arith.constant 1 : i32
      %4 = arith.index_cast %c1_i32_5 : i32 to index
      %c1_6 = arith.constant 1 : index
      scf.for %arg3 = %c0_4 to %4 step %c1_6 {
        %c0_7 = arith.constant 0 : index
        %c1_i32_8 = arith.constant 1 : i32
        %5 = arith.index_cast %c1_i32_8 : i32 to index
        %c1_9 = arith.constant 1 : index
        scf.for %arg4 = %c0_7 to %5 step %c1_9 {
          %cst = arith.constant 0.000000e+00 : f32
          linalg.fill ins(%cst : f32) outs(%alloca_2 : memref<2x4x4xf32>)
          %cst_10 = arith.constant 0.000000e+00 : f32
          linalg.fill ins(%cst_10 : f32) outs(%alloca_1 : memref<2x4xf32>)
          %cst_11 = arith.constant 0.000000e+00 : f32
          linalg.fill ins(%cst_11 : f32) outs(%alloca_0 : memref<2x4xf32>)
          %cst_12 = arith.constant 0.000000e+00 : f32
          linalg.fill ins(%cst_12 : f32) outs(%alloca : memref<2x4xf32>)
          %c0_13 = arith.constant 0 : index
          %c2_i32_14 = arith.constant 2 : i32
          %6 = arith.index_cast %c2_i32_14 : i32 to index
          %c1_15 = arith.constant 1 : index
          scf.for %arg5 = %c0_13 to %6 step %c1_15 {
            %c0_108 = arith.constant 0 : index
            %c4_i32 = arith.constant 4 : i32
            %324 = arith.index_cast %c4_i32 : i32 to index
            %c1_109 = arith.constant 1 : index
            scf.for %arg6 = %c0_108 to %324 step %c1_109 {
              %c0_110 = arith.constant 0 : index
              %c4_i32_111 = arith.constant 4 : i32
              %325 = arith.index_cast %c4_i32_111 : i32 to index
              %c1_112 = arith.constant 1 : index
              scf.for %arg7 = %c0_110 to %325 step %c1_112 {
                %326 = arith.index_cast %arg2 : index to i32
                %c2_i32_113 = arith.constant 2 : i32
                %327 = arith.muli %326, %c2_i32_113 : i32
                %328 = arith.index_cast %327 : i32 to index
                %329 = arith.addi %arg5, %328 : index
                %true = arith.constant true
                %c2_i32_114 = arith.constant 2 : i32
                %330 = arith.index_cast %c2_i32_114 : i32 to index
                %c0_115 = arith.constant 0 : index
                %331 = arith.cmpi sge, %arg5, %c0_115 : index
                %332 = arith.andi %true, %331 : i1
                %333 = arith.cmpi slt, %arg5, %330 : index
                %334 = arith.andi %332, %333 : i1
                %c4_i32_116 = arith.constant 4 : i32
                %335 = arith.index_cast %c4_i32_116 : i32 to index
                %c0_117 = arith.constant 0 : index
                %336 = arith.cmpi sge, %arg6, %c0_117 : index
                %337 = arith.andi %334, %336 : i1
                %338 = arith.cmpi slt, %arg6, %335 : index
                %339 = arith.andi %337, %338 : i1
                %c4_i32_118 = arith.constant 4 : i32
                %340 = arith.index_cast %c4_i32_118 : i32 to index
                %c0_119 = arith.constant 0 : index
                %341 = arith.cmpi sge, %arg7, %c0_119 : index
                %342 = arith.andi %339, %341 : i1
                %343 = arith.cmpi slt, %arg7, %340 : index
                %344 = arith.andi %342, %343 : i1
                scf.if %344 {
                  %true_120 = arith.constant true
                  %c0_121 = arith.constant 0 : index
                  %345 = arith.cmpi sge, %329, %c0_121 : index
                  %346 = arith.andi %true_120, %345 : i1
                  %347 = arith.cmpi slt, %329, %dim : index
                  %348 = arith.andi %346, %347 : i1
                  %c4_i32_122 = arith.constant 4 : i32
                  %349 = arith.index_cast %c4_i32_122 : i32 to index
                  %c0_123 = arith.constant 0 : index
                  %350 = arith.cmpi sge, %arg6, %c0_123 : index
                  %351 = arith.andi %348, %350 : i1
                  %352 = arith.cmpi slt, %arg6, %349 : index
                  %353 = arith.andi %351, %352 : i1
                  %c4_i32_124 = arith.constant 4 : i32
                  %354 = arith.index_cast %c4_i32_124 : i32 to index
                  %c0_125 = arith.constant 0 : index
                  %355 = arith.cmpi sge, %arg7, %c0_125 : index
                  %356 = arith.andi %353, %355 : i1
                  %357 = arith.cmpi slt, %arg7, %354 : index
                  %358 = arith.andi %356, %357 : i1
                  %359 = scf.if %358 -> (f32) {
                    %360 = memref.load %arg0[%329, %arg6, %arg7] : memref<?x4x4xf32>
                    scf.yield %360 : f32
                  } else {
                    %cst_126 = arith.constant 0.000000e+00 : f32
                    scf.yield %cst_126 : f32
                  }
                  memref.store %359, %alloca_2[%arg5, %arg6, %arg7] : memref<2x4x4xf32>
                }
              }
            }
          }
          %c0_i32_16 = arith.constant 0 : i32
          %7 = arith.index_cast %c0_i32_16 : i32 to index
          %c1_i32_17 = arith.constant 1 : i32
          %8 = arith.index_cast %c1_i32_17 : i32 to index
          %c1_18 = arith.constant 1 : index
          %9 = arith.muli %c1_18, %8 : index
          %10 = arith.addi %7, %9 : index
          %c0_i32_19 = arith.constant 0 : i32
          %11 = arith.index_cast %c0_i32_19 : i32 to index
          %c1_i32_20 = arith.constant 1 : i32
          %12 = arith.index_cast %c1_i32_20 : i32 to index
          %c1_21 = arith.constant 1 : index
          %13 = arith.muli %c1_21, %12 : index
          %14 = arith.addi %11, %13 : index
          %c2 = arith.constant 2 : index
          %15 = arith.muli %c2, %12 : index
          %16 = arith.addi %11, %15 : index
          %c3 = arith.constant 3 : index
          %17 = arith.muli %c3, %12 : index
          %18 = arith.addi %11, %17 : index
          %c0_i32_22 = arith.constant 0 : i32
          %19 = arith.index_cast %c0_i32_22 : i32 to index
          %c1_i32_23 = arith.constant 1 : i32
          %20 = arith.index_cast %c1_i32_23 : i32 to index
          %c1_24 = arith.constant 1 : index
          %21 = arith.muli %c1_24, %20 : index
          %22 = arith.addi %19, %21 : index
          %c2_25 = arith.constant 2 : index
          %23 = arith.muli %c2_25, %20 : index
          %24 = arith.addi %19, %23 : index
          %c3_26 = arith.constant 3 : index
          %25 = arith.muli %c3_26, %20 : index
          %26 = arith.addi %19, %25 : index
          %c0_i32_27 = arith.constant 0 : i32
          %27 = arith.index_cast %c0_i32_27 : i32 to index
          %c1_i32_28 = arith.constant 1 : i32
          %28 = arith.index_cast %c1_i32_28 : i32 to index
          %c1_29 = arith.constant 1 : index
          %29 = arith.muli %c1_29, %28 : index
          %30 = arith.addi %27, %29 : index
          %c0_i32_30 = arith.constant 0 : i32
          %31 = arith.index_cast %c0_i32_30 : i32 to index
          %c1_i32_31 = arith.constant 1 : i32
          %32 = arith.index_cast %c1_i32_31 : i32 to index
          %c1_32 = arith.constant 1 : index
          %33 = arith.muli %c1_32, %32 : index
          %34 = arith.addi %31, %33 : index
          %c2_33 = arith.constant 2 : index
          %35 = arith.muli %c2_33, %32 : index
          %36 = arith.addi %31, %35 : index
          %c3_34 = arith.constant 3 : index
          %37 = arith.muli %c3_34, %32 : index
          %38 = arith.addi %31, %37 : index
          %39 = memref.load %alloca_2[%7, %11, %19] : memref<2x4x4xf32>
          %40 = memref.load %alloca_2[%7, %11, %22] : memref<2x4x4xf32>
          %41 = arith.cmpf ogt, %39, %40 : f32
          %42 = arith.select %41, %39, %40 : f32
          %43 = memref.load %alloca_2[%7, %11, %24] : memref<2x4x4xf32>
          %44 = arith.cmpf ogt, %42, %43 : f32
          %45 = arith.select %44, %42, %43 : f32
          %46 = memref.load %alloca_2[%7, %11, %26] : memref<2x4x4xf32>
          %47 = arith.cmpf ogt, %45, %46 : f32
          %48 = arith.select %47, %45, %46 : f32
          memref.store %48, %alloca[%27, %31] : memref<2x4xf32>
          %49 = memref.load %alloca_2[%7, %14, %19] : memref<2x4x4xf32>
          %50 = memref.load %alloca_2[%7, %14, %22] : memref<2x4x4xf32>
          %51 = arith.cmpf ogt, %49, %50 : f32
          %52 = arith.select %51, %49, %50 : f32
          %53 = memref.load %alloca_2[%7, %14, %24] : memref<2x4x4xf32>
          %54 = arith.cmpf ogt, %52, %53 : f32
          %55 = arith.select %54, %52, %53 : f32
          %56 = memref.load %alloca_2[%7, %14, %26] : memref<2x4x4xf32>
          %57 = arith.cmpf ogt, %55, %56 : f32
          %58 = arith.select %57, %55, %56 : f32
          memref.store %58, %alloca[%27, %34] : memref<2x4xf32>
          %59 = memref.load %alloca_2[%7, %16, %19] : memref<2x4x4xf32>
          %60 = memref.load %alloca_2[%7, %16, %22] : memref<2x4x4xf32>
          %61 = arith.cmpf ogt, %59, %60 : f32
          %62 = arith.select %61, %59, %60 : f32
          %63 = memref.load %alloca_2[%7, %16, %24] : memref<2x4x4xf32>
          %64 = arith.cmpf ogt, %62, %63 : f32
          %65 = arith.select %64, %62, %63 : f32
          %66 = memref.load %alloca_2[%7, %16, %26] : memref<2x4x4xf32>
          %67 = arith.cmpf ogt, %65, %66 : f32
          %68 = arith.select %67, %65, %66 : f32
          memref.store %68, %alloca[%27, %36] : memref<2x4xf32>
          %69 = memref.load %alloca_2[%7, %18, %19] : memref<2x4x4xf32>
          %70 = memref.load %alloca_2[%7, %18, %22] : memref<2x4x4xf32>
          %71 = arith.cmpf ogt, %69, %70 : f32
          %72 = arith.select %71, %69, %70 : f32
          %73 = memref.load %alloca_2[%7, %18, %24] : memref<2x4x4xf32>
          %74 = arith.cmpf ogt, %72, %73 : f32
          %75 = arith.select %74, %72, %73 : f32
          %76 = memref.load %alloca_2[%7, %18, %26] : memref<2x4x4xf32>
          %77 = arith.cmpf ogt, %75, %76 : f32
          %78 = arith.select %77, %75, %76 : f32
          memref.store %78, %alloca[%27, %38] : memref<2x4xf32>
          %79 = memref.load %alloca_2[%10, %11, %19] : memref<2x4x4xf32>
          %80 = memref.load %alloca_2[%10, %11, %22] : memref<2x4x4xf32>
          %81 = arith.cmpf ogt, %79, %80 : f32
          %82 = arith.select %81, %79, %80 : f32
          %83 = memref.load %alloca_2[%10, %11, %24] : memref<2x4x4xf32>
          %84 = arith.cmpf ogt, %82, %83 : f32
          %85 = arith.select %84, %82, %83 : f32
          %86 = memref.load %alloca_2[%10, %11, %26] : memref<2x4x4xf32>
          %87 = arith.cmpf ogt, %85, %86 : f32
          %88 = arith.select %87, %85, %86 : f32
          memref.store %88, %alloca[%30, %31] : memref<2x4xf32>
          %89 = memref.load %alloca_2[%10, %14, %19] : memref<2x4x4xf32>
          %90 = memref.load %alloca_2[%10, %14, %22] : memref<2x4x4xf32>
          %91 = arith.cmpf ogt, %89, %90 : f32
          %92 = arith.select %91, %89, %90 : f32
          %93 = memref.load %alloca_2[%10, %14, %24] : memref<2x4x4xf32>
          %94 = arith.cmpf ogt, %92, %93 : f32
          %95 = arith.select %94, %92, %93 : f32
          %96 = memref.load %alloca_2[%10, %14, %26] : memref<2x4x4xf32>
          %97 = arith.cmpf ogt, %95, %96 : f32
          %98 = arith.select %97, %95, %96 : f32
          memref.store %98, %alloca[%30, %34] : memref<2x4xf32>
          %99 = memref.load %alloca_2[%10, %16, %19] : memref<2x4x4xf32>
          %100 = memref.load %alloca_2[%10, %16, %22] : memref<2x4x4xf32>
          %101 = arith.cmpf ogt, %99, %100 : f32
          %102 = arith.select %101, %99, %100 : f32
          %103 = memref.load %alloca_2[%10, %16, %24] : memref<2x4x4xf32>
          %104 = arith.cmpf ogt, %102, %103 : f32
          %105 = arith.select %104, %102, %103 : f32
          %106 = memref.load %alloca_2[%10, %16, %26] : memref<2x4x4xf32>
          %107 = arith.cmpf ogt, %105, %106 : f32
          %108 = arith.select %107, %105, %106 : f32
          memref.store %108, %alloca[%30, %36] : memref<2x4xf32>
          %109 = memref.load %alloca_2[%10, %18, %19] : memref<2x4x4xf32>
          %110 = memref.load %alloca_2[%10, %18, %22] : memref<2x4x4xf32>
          %111 = arith.cmpf ogt, %109, %110 : f32
          %112 = arith.select %111, %109, %110 : f32
          %113 = memref.load %alloca_2[%10, %18, %24] : memref<2x4x4xf32>
          %114 = arith.cmpf ogt, %112, %113 : f32
          %115 = arith.select %114, %112, %113 : f32
          %116 = memref.load %alloca_2[%10, %18, %26] : memref<2x4x4xf32>
          %117 = arith.cmpf ogt, %115, %116 : f32
          %118 = arith.select %117, %115, %116 : f32
          memref.store %118, %alloca[%30, %38] : memref<2x4xf32>
          %c0_i32_35 = arith.constant 0 : i32
          %119 = arith.index_cast %c0_i32_35 : i32 to index
          %c2_i32_36 = arith.constant 2 : i32
          %120 = arith.index_cast %c2_i32_36 : i32 to index
          %c1_37 = arith.constant 1 : index
          %121 = arith.addi %119, %120 : index
          scf.parallel (%arg5) = (%119) to (%121) step (%c1_37) {
            %c0_i32_108 = arith.constant 0 : i32
            %324 = arith.index_cast %c0_i32_108 : i32 to index
            %c4_i32 = arith.constant 4 : i32
            %325 = arith.index_cast %c4_i32 : i32 to index
            %c1_109 = arith.constant 1 : index
            %326 = arith.addi %324, %325 : index
            scf.parallel (%arg6) = (%324) to (%326) step (%c1_109) {
              %c0_i32_110 = arith.constant 0 : i32
              %327 = arith.index_cast %c0_i32_110 : i32 to index
              %c4_i32_111 = arith.constant 4 : i32
              %328 = arith.index_cast %c4_i32_111 : i32 to index
              %c1_112 = arith.constant 1 : index
              %329 = arith.addi %327, %328 : index
              scf.parallel (%arg7) = (%327) to (%329) step (%c1_112) {
                %330 = memref.load %alloca_2[%arg5, %arg6, %arg7] : memref<2x4x4xf32>
                %331 = memref.load %alloca[%arg5, %arg6] : memref<2x4xf32>
                %332 = arith.subf %330, %331 : f32
                %333 = math.exp %332 : f32
                memref.store %333, %alloca_2[%arg5, %arg6, %arg7] : memref<2x4x4xf32>
                scf.reduce 
              }
              scf.reduce 
            }
            scf.reduce 
          }
          %c0_i32_38 = arith.constant 0 : i32
          %122 = arith.index_cast %c0_i32_38 : i32 to index
          %c1_i32_39 = arith.constant 1 : i32
          %123 = arith.index_cast %c1_i32_39 : i32 to index
          %c1_40 = arith.constant 1 : index
          %124 = arith.muli %c1_40, %123 : index
          %125 = arith.addi %122, %124 : index
          %c0_i32_41 = arith.constant 0 : i32
          %126 = arith.index_cast %c0_i32_41 : i32 to index
          %c1_i32_42 = arith.constant 1 : i32
          %127 = arith.index_cast %c1_i32_42 : i32 to index
          %c1_43 = arith.constant 1 : index
          %128 = arith.muli %c1_43, %127 : index
          %129 = arith.addi %126, %128 : index
          %c2_44 = arith.constant 2 : index
          %130 = arith.muli %c2_44, %127 : index
          %131 = arith.addi %126, %130 : index
          %c3_45 = arith.constant 3 : index
          %132 = arith.muli %c3_45, %127 : index
          %133 = arith.addi %126, %132 : index
          %c0_i32_46 = arith.constant 0 : i32
          %134 = arith.index_cast %c0_i32_46 : i32 to index
          %c1_i32_47 = arith.constant 1 : i32
          %135 = arith.index_cast %c1_i32_47 : i32 to index
          %c1_48 = arith.constant 1 : index
          %136 = arith.muli %c1_48, %135 : index
          %137 = arith.addi %134, %136 : index
          %c2_49 = arith.constant 2 : index
          %138 = arith.muli %c2_49, %135 : index
          %139 = arith.addi %134, %138 : index
          %c3_50 = arith.constant 3 : index
          %140 = arith.muli %c3_50, %135 : index
          %141 = arith.addi %134, %140 : index
          %c0_i32_51 = arith.constant 0 : i32
          %142 = arith.index_cast %c0_i32_51 : i32 to index
          %c1_i32_52 = arith.constant 1 : i32
          %143 = arith.index_cast %c1_i32_52 : i32 to index
          %c1_53 = arith.constant 1 : index
          %144 = arith.muli %c1_53, %143 : index
          %145 = arith.addi %142, %144 : index
          %c0_i32_54 = arith.constant 0 : i32
          %146 = arith.index_cast %c0_i32_54 : i32 to index
          %c1_i32_55 = arith.constant 1 : i32
          %147 = arith.index_cast %c1_i32_55 : i32 to index
          %c1_56 = arith.constant 1 : index
          %148 = arith.muli %c1_56, %147 : index
          %149 = arith.addi %146, %148 : index
          %c2_57 = arith.constant 2 : index
          %150 = arith.muli %c2_57, %147 : index
          %151 = arith.addi %146, %150 : index
          %c3_58 = arith.constant 3 : index
          %152 = arith.muli %c3_58, %147 : index
          %153 = arith.addi %146, %152 : index
          %cst_59 = arith.constant 0.000000e+00 : f32
          %154 = memref.load %alloca_2[%122, %126, %134] : memref<2x4x4xf32>
          %155 = arith.addf %cst_59, %154 : f32
          %156 = memref.load %alloca_2[%122, %126, %137] : memref<2x4x4xf32>
          %157 = arith.addf %155, %156 : f32
          %158 = memref.load %alloca_2[%122, %126, %139] : memref<2x4x4xf32>
          %159 = arith.addf %157, %158 : f32
          %160 = memref.load %alloca_2[%122, %126, %141] : memref<2x4x4xf32>
          %161 = arith.addf %159, %160 : f32
          memref.store %161, %alloca_1[%142, %146] : memref<2x4xf32>
          %cst_60 = arith.constant 0.000000e+00 : f32
          %162 = memref.load %alloca_2[%122, %129, %134] : memref<2x4x4xf32>
          %163 = arith.addf %cst_60, %162 : f32
          %164 = memref.load %alloca_2[%122, %129, %137] : memref<2x4x4xf32>
          %165 = arith.addf %163, %164 : f32
          %166 = memref.load %alloca_2[%122, %129, %139] : memref<2x4x4xf32>
          %167 = arith.addf %165, %166 : f32
          %168 = memref.load %alloca_2[%122, %129, %141] : memref<2x4x4xf32>
          %169 = arith.addf %167, %168 : f32
          memref.store %169, %alloca_1[%142, %149] : memref<2x4xf32>
          %cst_61 = arith.constant 0.000000e+00 : f32
          %170 = memref.load %alloca_2[%122, %131, %134] : memref<2x4x4xf32>
          %171 = arith.addf %cst_61, %170 : f32
          %172 = memref.load %alloca_2[%122, %131, %137] : memref<2x4x4xf32>
          %173 = arith.addf %171, %172 : f32
          %174 = memref.load %alloca_2[%122, %131, %139] : memref<2x4x4xf32>
          %175 = arith.addf %173, %174 : f32
          %176 = memref.load %alloca_2[%122, %131, %141] : memref<2x4x4xf32>
          %177 = arith.addf %175, %176 : f32
          memref.store %177, %alloca_1[%142, %151] : memref<2x4xf32>
          %cst_62 = arith.constant 0.000000e+00 : f32
          %178 = memref.load %alloca_2[%122, %133, %134] : memref<2x4x4xf32>
          %179 = arith.addf %cst_62, %178 : f32
          %180 = memref.load %alloca_2[%122, %133, %137] : memref<2x4x4xf32>
          %181 = arith.addf %179, %180 : f32
          %182 = memref.load %alloca_2[%122, %133, %139] : memref<2x4x4xf32>
          %183 = arith.addf %181, %182 : f32
          %184 = memref.load %alloca_2[%122, %133, %141] : memref<2x4x4xf32>
          %185 = arith.addf %183, %184 : f32
          memref.store %185, %alloca_1[%142, %153] : memref<2x4xf32>
          %cst_63 = arith.constant 0.000000e+00 : f32
          %186 = memref.load %alloca_2[%125, %126, %134] : memref<2x4x4xf32>
          %187 = arith.addf %cst_63, %186 : f32
          %188 = memref.load %alloca_2[%125, %126, %137] : memref<2x4x4xf32>
          %189 = arith.addf %187, %188 : f32
          %190 = memref.load %alloca_2[%125, %126, %139] : memref<2x4x4xf32>
          %191 = arith.addf %189, %190 : f32
          %192 = memref.load %alloca_2[%125, %126, %141] : memref<2x4x4xf32>
          %193 = arith.addf %191, %192 : f32
          memref.store %193, %alloca_1[%145, %146] : memref<2x4xf32>
          %cst_64 = arith.constant 0.000000e+00 : f32
          %194 = memref.load %alloca_2[%125, %129, %134] : memref<2x4x4xf32>
          %195 = arith.addf %cst_64, %194 : f32
          %196 = memref.load %alloca_2[%125, %129, %137] : memref<2x4x4xf32>
          %197 = arith.addf %195, %196 : f32
          %198 = memref.load %alloca_2[%125, %129, %139] : memref<2x4x4xf32>
          %199 = arith.addf %197, %198 : f32
          %200 = memref.load %alloca_2[%125, %129, %141] : memref<2x4x4xf32>
          %201 = arith.addf %199, %200 : f32
          memref.store %201, %alloca_1[%145, %149] : memref<2x4xf32>
          %cst_65 = arith.constant 0.000000e+00 : f32
          %202 = memref.load %alloca_2[%125, %131, %134] : memref<2x4x4xf32>
          %203 = arith.addf %cst_65, %202 : f32
          %204 = memref.load %alloca_2[%125, %131, %137] : memref<2x4x4xf32>
          %205 = arith.addf %203, %204 : f32
          %206 = memref.load %alloca_2[%125, %131, %139] : memref<2x4x4xf32>
          %207 = arith.addf %205, %206 : f32
          %208 = memref.load %alloca_2[%125, %131, %141] : memref<2x4x4xf32>
          %209 = arith.addf %207, %208 : f32
          memref.store %209, %alloca_1[%145, %151] : memref<2x4xf32>
          %cst_66 = arith.constant 0.000000e+00 : f32
          %210 = memref.load %alloca_2[%125, %133, %134] : memref<2x4x4xf32>
          %211 = arith.addf %cst_66, %210 : f32
          %212 = memref.load %alloca_2[%125, %133, %137] : memref<2x4x4xf32>
          %213 = arith.addf %211, %212 : f32
          %214 = memref.load %alloca_2[%125, %133, %139] : memref<2x4x4xf32>
          %215 = arith.addf %213, %214 : f32
          %216 = memref.load %alloca_2[%125, %133, %141] : memref<2x4x4xf32>
          %217 = arith.addf %215, %216 : f32
          memref.store %217, %alloca_1[%145, %153] : memref<2x4xf32>
          %c0_i32_67 = arith.constant 0 : i32
          %218 = arith.index_cast %c0_i32_67 : i32 to index
          %c2_i32_68 = arith.constant 2 : i32
          %219 = arith.index_cast %c2_i32_68 : i32 to index
          %c1_69 = arith.constant 1 : index
          %220 = arith.addi %218, %219 : index
          scf.parallel (%arg5) = (%218) to (%220) step (%c1_69) {
            %c0_i32_108 = arith.constant 0 : i32
            %324 = arith.index_cast %c0_i32_108 : i32 to index
            %c4_i32 = arith.constant 4 : i32
            %325 = arith.index_cast %c4_i32 : i32 to index
            %c1_109 = arith.constant 1 : index
            %326 = arith.addi %324, %325 : index
            scf.parallel (%arg6) = (%324) to (%326) step (%c1_109) {
              %c0_i32_110 = arith.constant 0 : i32
              %327 = arith.index_cast %c0_i32_110 : i32 to index
              %c4_i32_111 = arith.constant 4 : i32
              %328 = arith.index_cast %c4_i32_111 : i32 to index
              %c1_112 = arith.constant 1 : index
              %329 = arith.addi %327, %328 : index
              scf.parallel (%arg7) = (%327) to (%329) step (%c1_112) {
                %330 = memref.load %alloca_2[%arg5, %arg6, %arg7] : memref<2x4x4xf32>
                %331 = memref.load %alloca_1[%arg5, %arg6] : memref<2x4xf32>
                %332 = arith.divf %330, %331 : f32
                %cst_113 = arith.constant 9.99999997E-7 : f32
                %333 = arith.addf %332, %cst_113 : f32
                memref.store %333, %alloca_2[%arg5, %arg6, %arg7] : memref<2x4x4xf32>
                scf.reduce 
              }
              scf.reduce 
            }
            scf.reduce 
          }
          %c0_i32_70 = arith.constant 0 : i32
          %221 = arith.index_cast %c0_i32_70 : i32 to index
          %c1_i32_71 = arith.constant 1 : i32
          %222 = arith.index_cast %c1_i32_71 : i32 to index
          %c1_72 = arith.constant 1 : index
          %223 = arith.muli %c1_72, %222 : index
          %224 = arith.addi %221, %223 : index
          %c0_i32_73 = arith.constant 0 : i32
          %225 = arith.index_cast %c0_i32_73 : i32 to index
          %c1_i32_74 = arith.constant 1 : i32
          %226 = arith.index_cast %c1_i32_74 : i32 to index
          %c1_75 = arith.constant 1 : index
          %227 = arith.muli %c1_75, %226 : index
          %228 = arith.addi %225, %227 : index
          %c2_76 = arith.constant 2 : index
          %229 = arith.muli %c2_76, %226 : index
          %230 = arith.addi %225, %229 : index
          %c3_77 = arith.constant 3 : index
          %231 = arith.muli %c3_77, %226 : index
          %232 = arith.addi %225, %231 : index
          %c0_i32_78 = arith.constant 0 : i32
          %233 = arith.index_cast %c0_i32_78 : i32 to index
          %c1_i32_79 = arith.constant 1 : i32
          %234 = arith.index_cast %c1_i32_79 : i32 to index
          %c1_80 = arith.constant 1 : index
          %235 = arith.muli %c1_80, %234 : index
          %236 = arith.addi %233, %235 : index
          %c2_81 = arith.constant 2 : index
          %237 = arith.muli %c2_81, %234 : index
          %238 = arith.addi %233, %237 : index
          %c3_82 = arith.constant 3 : index
          %239 = arith.muli %c3_82, %234 : index
          %240 = arith.addi %233, %239 : index
          %c0_i32_83 = arith.constant 0 : i32
          %241 = arith.index_cast %c0_i32_83 : i32 to index
          %c1_i32_84 = arith.constant 1 : i32
          %242 = arith.index_cast %c1_i32_84 : i32 to index
          %c1_85 = arith.constant 1 : index
          %243 = arith.muli %c1_85, %242 : index
          %244 = arith.addi %241, %243 : index
          %c0_i32_86 = arith.constant 0 : i32
          %245 = arith.index_cast %c0_i32_86 : i32 to index
          %c1_i32_87 = arith.constant 1 : i32
          %246 = arith.index_cast %c1_i32_87 : i32 to index
          %c1_88 = arith.constant 1 : index
          %247 = arith.muli %c1_88, %246 : index
          %248 = arith.addi %245, %247 : index
          %c2_89 = arith.constant 2 : index
          %249 = arith.muli %c2_89, %246 : index
          %250 = arith.addi %245, %249 : index
          %c3_90 = arith.constant 3 : index
          %251 = arith.muli %c3_90, %246 : index
          %252 = arith.addi %245, %251 : index
          %cst_91 = arith.constant 0.000000e+00 : f32
          %253 = memref.load %alloca_2[%221, %225, %233] : memref<2x4x4xf32>
          %254 = arith.addf %cst_91, %253 : f32
          %255 = memref.load %alloca_2[%221, %228, %233] : memref<2x4x4xf32>
          %256 = arith.addf %254, %255 : f32
          %257 = memref.load %alloca_2[%221, %230, %233] : memref<2x4x4xf32>
          %258 = arith.addf %256, %257 : f32
          %259 = memref.load %alloca_2[%221, %232, %233] : memref<2x4x4xf32>
          %260 = arith.addf %258, %259 : f32
          memref.store %260, %alloca_0[%241, %245] : memref<2x4xf32>
          %cst_92 = arith.constant 0.000000e+00 : f32
          %261 = memref.load %alloca_2[%221, %225, %236] : memref<2x4x4xf32>
          %262 = arith.addf %cst_92, %261 : f32
          %263 = memref.load %alloca_2[%221, %228, %236] : memref<2x4x4xf32>
          %264 = arith.addf %262, %263 : f32
          %265 = memref.load %alloca_2[%221, %230, %236] : memref<2x4x4xf32>
          %266 = arith.addf %264, %265 : f32
          %267 = memref.load %alloca_2[%221, %232, %236] : memref<2x4x4xf32>
          %268 = arith.addf %266, %267 : f32
          memref.store %268, %alloca_0[%241, %248] : memref<2x4xf32>
          %cst_93 = arith.constant 0.000000e+00 : f32
          %269 = memref.load %alloca_2[%221, %225, %238] : memref<2x4x4xf32>
          %270 = arith.addf %cst_93, %269 : f32
          %271 = memref.load %alloca_2[%221, %228, %238] : memref<2x4x4xf32>
          %272 = arith.addf %270, %271 : f32
          %273 = memref.load %alloca_2[%221, %230, %238] : memref<2x4x4xf32>
          %274 = arith.addf %272, %273 : f32
          %275 = memref.load %alloca_2[%221, %232, %238] : memref<2x4x4xf32>
          %276 = arith.addf %274, %275 : f32
          memref.store %276, %alloca_0[%241, %250] : memref<2x4xf32>
          %cst_94 = arith.constant 0.000000e+00 : f32
          %277 = memref.load %alloca_2[%221, %225, %240] : memref<2x4x4xf32>
          %278 = arith.addf %cst_94, %277 : f32
          %279 = memref.load %alloca_2[%221, %228, %240] : memref<2x4x4xf32>
          %280 = arith.addf %278, %279 : f32
          %281 = memref.load %alloca_2[%221, %230, %240] : memref<2x4x4xf32>
          %282 = arith.addf %280, %281 : f32
          %283 = memref.load %alloca_2[%221, %232, %240] : memref<2x4x4xf32>
          %284 = arith.addf %282, %283 : f32
          memref.store %284, %alloca_0[%241, %252] : memref<2x4xf32>
          %cst_95 = arith.constant 0.000000e+00 : f32
          %285 = memref.load %alloca_2[%224, %225, %233] : memref<2x4x4xf32>
          %286 = arith.addf %cst_95, %285 : f32
          %287 = memref.load %alloca_2[%224, %228, %233] : memref<2x4x4xf32>
          %288 = arith.addf %286, %287 : f32
          %289 = memref.load %alloca_2[%224, %230, %233] : memref<2x4x4xf32>
          %290 = arith.addf %288, %289 : f32
          %291 = memref.load %alloca_2[%224, %232, %233] : memref<2x4x4xf32>
          %292 = arith.addf %290, %291 : f32
          memref.store %292, %alloca_0[%244, %245] : memref<2x4xf32>
          %cst_96 = arith.constant 0.000000e+00 : f32
          %293 = memref.load %alloca_2[%224, %225, %236] : memref<2x4x4xf32>
          %294 = arith.addf %cst_96, %293 : f32
          %295 = memref.load %alloca_2[%224, %228, %236] : memref<2x4x4xf32>
          %296 = arith.addf %294, %295 : f32
          %297 = memref.load %alloca_2[%224, %230, %236] : memref<2x4x4xf32>
          %298 = arith.addf %296, %297 : f32
          %299 = memref.load %alloca_2[%224, %232, %236] : memref<2x4x4xf32>
          %300 = arith.addf %298, %299 : f32
          memref.store %300, %alloca_0[%244, %248] : memref<2x4xf32>
          %cst_97 = arith.constant 0.000000e+00 : f32
          %301 = memref.load %alloca_2[%224, %225, %238] : memref<2x4x4xf32>
          %302 = arith.addf %cst_97, %301 : f32
          %303 = memref.load %alloca_2[%224, %228, %238] : memref<2x4x4xf32>
          %304 = arith.addf %302, %303 : f32
          %305 = memref.load %alloca_2[%224, %230, %238] : memref<2x4x4xf32>
          %306 = arith.addf %304, %305 : f32
          %307 = memref.load %alloca_2[%224, %232, %238] : memref<2x4x4xf32>
          %308 = arith.addf %306, %307 : f32
          memref.store %308, %alloca_0[%244, %250] : memref<2x4xf32>
          %cst_98 = arith.constant 0.000000e+00 : f32
          %309 = memref.load %alloca_2[%224, %225, %240] : memref<2x4x4xf32>
          %310 = arith.addf %cst_98, %309 : f32
          %311 = memref.load %alloca_2[%224, %228, %240] : memref<2x4x4xf32>
          %312 = arith.addf %310, %311 : f32
          %313 = memref.load %alloca_2[%224, %230, %240] : memref<2x4x4xf32>
          %314 = arith.addf %312, %313 : f32
          %315 = memref.load %alloca_2[%224, %232, %240] : memref<2x4x4xf32>
          %316 = arith.addf %314, %315 : f32
          memref.store %316, %alloca_0[%244, %252] : memref<2x4xf32>
          %c0_i32_99 = arith.constant 0 : i32
          %317 = arith.index_cast %c0_i32_99 : i32 to index
          %c2_i32_100 = arith.constant 2 : i32
          %318 = arith.index_cast %c2_i32_100 : i32 to index
          %c1_101 = arith.constant 1 : index
          %319 = arith.addi %317, %318 : index
          scf.parallel (%arg5) = (%317) to (%319) step (%c1_101) {
            %c0_i32_108 = arith.constant 0 : i32
            %324 = arith.index_cast %c0_i32_108 : i32 to index
            %c4_i32 = arith.constant 4 : i32
            %325 = arith.index_cast %c4_i32 : i32 to index
            %c1_109 = arith.constant 1 : index
            %326 = arith.addi %324, %325 : index
            scf.parallel (%arg6) = (%324) to (%326) step (%c1_109) {
              %c0_i32_110 = arith.constant 0 : i32
              %327 = arith.index_cast %c0_i32_110 : i32 to index
              %c4_i32_111 = arith.constant 4 : i32
              %328 = arith.index_cast %c4_i32_111 : i32 to index
              %c1_112 = arith.constant 1 : index
              %329 = arith.addi %327, %328 : index
              scf.parallel (%arg7) = (%327) to (%329) step (%c1_112) {
                %330 = memref.load %alloca_2[%arg5, %arg6, %arg7] : memref<2x4x4xf32>
                %331 = memref.load %alloca_0[%arg5, %arg7] : memref<2x4xf32>
                %cst_113 = arith.constant 9.99999997E-7 : f32
                %332 = arith.addf %331, %cst_113 : f32
                %333 = arith.divf %330, %332 : f32
                memref.store %333, %alloca_2[%arg5, %arg6, %arg7] : memref<2x4x4xf32>
                scf.reduce 
              }
              scf.reduce 
            }
            scf.reduce 
          }
          %c0_i32_102 = arith.constant 0 : i32
          %320 = arith.index_cast %c0_i32_102 : i32 to index
          %c2_i32_103 = arith.constant 2 : i32
          %321 = arith.index_cast %c2_i32_103 : i32 to index
          %c1_104 = arith.constant 1 : index
          %322 = arith.addi %320, %321 : index
          scf.for %arg5 = %320 to %322 step %c1_104 {
            %c0_i32_108 = arith.constant 0 : i32
            %324 = arith.index_cast %c0_i32_108 : i32 to index
            %c1_i32_109 = arith.constant 1 : i32
            %325 = arith.index_cast %c1_i32_109 : i32 to index
            %c1_110 = arith.constant 1 : index
            %326 = arith.muli %c1_110, %325 : index
            %327 = arith.addi %324, %326 : index
            %c0_i32_111 = arith.constant 0 : i32
            %328 = arith.index_cast %c0_i32_111 : i32 to index
            %c1_i32_112 = arith.constant 1 : i32
            %329 = arith.index_cast %c1_i32_112 : i32 to index
            %c1_113 = arith.constant 1 : index
            %330 = arith.muli %c1_113, %329 : index
            %331 = arith.addi %328, %330 : index
            %c2_114 = arith.constant 2 : index
            %332 = arith.muli %c2_114, %329 : index
            %333 = arith.addi %328, %332 : index
            %c3_115 = arith.constant 3 : index
            %334 = arith.muli %c3_115, %329 : index
            %335 = arith.addi %328, %334 : index
            %c0_i32_116 = arith.constant 0 : i32
            %336 = arith.index_cast %c0_i32_116 : i32 to index
            %c1_i32_117 = arith.constant 1 : i32
            %337 = arith.index_cast %c1_i32_117 : i32 to index
            %c1_118 = arith.constant 1 : index
            %338 = arith.muli %c1_118, %337 : index
            %339 = arith.addi %336, %338 : index
            %c2_119 = arith.constant 2 : index
            %340 = arith.muli %c2_119, %337 : index
            %341 = arith.addi %336, %340 : index
            %c3_120 = arith.constant 3 : index
            %342 = arith.muli %c3_120, %337 : index
            %343 = arith.addi %336, %342 : index
            %c0_i32_121 = arith.constant 0 : i32
            %344 = arith.index_cast %c0_i32_121 : i32 to index
            %c1_i32_122 = arith.constant 1 : i32
            %345 = arith.index_cast %c1_i32_122 : i32 to index
            %c1_123 = arith.constant 1 : index
            %346 = arith.muli %c1_123, %345 : index
            %347 = arith.addi %344, %346 : index
            %c0_i32_124 = arith.constant 0 : i32
            %348 = arith.index_cast %c0_i32_124 : i32 to index
            %c1_i32_125 = arith.constant 1 : i32
            %349 = arith.index_cast %c1_i32_125 : i32 to index
            %c1_126 = arith.constant 1 : index
            %350 = arith.muli %c1_126, %349 : index
            %351 = arith.addi %348, %350 : index
            %c2_127 = arith.constant 2 : index
            %352 = arith.muli %c2_127, %349 : index
            %353 = arith.addi %348, %352 : index
            %c3_128 = arith.constant 3 : index
            %354 = arith.muli %c3_128, %349 : index
            %355 = arith.addi %348, %354 : index
            %cst_129 = arith.constant 0.000000e+00 : f32
            %356 = memref.load %alloca_2[%324, %328, %336] : memref<2x4x4xf32>
            %357 = arith.addf %cst_129, %356 : f32
            %358 = memref.load %alloca_2[%324, %328, %339] : memref<2x4x4xf32>
            %359 = arith.addf %357, %358 : f32
            %360 = memref.load %alloca_2[%324, %328, %341] : memref<2x4x4xf32>
            %361 = arith.addf %359, %360 : f32
            %362 = memref.load %alloca_2[%324, %328, %343] : memref<2x4x4xf32>
            %363 = arith.addf %361, %362 : f32
            memref.store %363, %alloca_1[%344, %348] : memref<2x4xf32>
            %cst_130 = arith.constant 0.000000e+00 : f32
            %364 = memref.load %alloca_2[%324, %331, %336] : memref<2x4x4xf32>
            %365 = arith.addf %cst_130, %364 : f32
            %366 = memref.load %alloca_2[%324, %331, %339] : memref<2x4x4xf32>
            %367 = arith.addf %365, %366 : f32
            %368 = memref.load %alloca_2[%324, %331, %341] : memref<2x4x4xf32>
            %369 = arith.addf %367, %368 : f32
            %370 = memref.load %alloca_2[%324, %331, %343] : memref<2x4x4xf32>
            %371 = arith.addf %369, %370 : f32
            memref.store %371, %alloca_1[%344, %351] : memref<2x4xf32>
            %cst_131 = arith.constant 0.000000e+00 : f32
            %372 = memref.load %alloca_2[%324, %333, %336] : memref<2x4x4xf32>
            %373 = arith.addf %cst_131, %372 : f32
            %374 = memref.load %alloca_2[%324, %333, %339] : memref<2x4x4xf32>
            %375 = arith.addf %373, %374 : f32
            %376 = memref.load %alloca_2[%324, %333, %341] : memref<2x4x4xf32>
            %377 = arith.addf %375, %376 : f32
            %378 = memref.load %alloca_2[%324, %333, %343] : memref<2x4x4xf32>
            %379 = arith.addf %377, %378 : f32
            memref.store %379, %alloca_1[%344, %353] : memref<2x4xf32>
            %cst_132 = arith.constant 0.000000e+00 : f32
            %380 = memref.load %alloca_2[%324, %335, %336] : memref<2x4x4xf32>
            %381 = arith.addf %cst_132, %380 : f32
            %382 = memref.load %alloca_2[%324, %335, %339] : memref<2x4x4xf32>
            %383 = arith.addf %381, %382 : f32
            %384 = memref.load %alloca_2[%324, %335, %341] : memref<2x4x4xf32>
            %385 = arith.addf %383, %384 : f32
            %386 = memref.load %alloca_2[%324, %335, %343] : memref<2x4x4xf32>
            %387 = arith.addf %385, %386 : f32
            memref.store %387, %alloca_1[%344, %355] : memref<2x4xf32>
            %cst_133 = arith.constant 0.000000e+00 : f32
            %388 = memref.load %alloca_2[%327, %328, %336] : memref<2x4x4xf32>
            %389 = arith.addf %cst_133, %388 : f32
            %390 = memref.load %alloca_2[%327, %328, %339] : memref<2x4x4xf32>
            %391 = arith.addf %389, %390 : f32
            %392 = memref.load %alloca_2[%327, %328, %341] : memref<2x4x4xf32>
            %393 = arith.addf %391, %392 : f32
            %394 = memref.load %alloca_2[%327, %328, %343] : memref<2x4x4xf32>
            %395 = arith.addf %393, %394 : f32
            memref.store %395, %alloca_1[%347, %348] : memref<2x4xf32>
            %cst_134 = arith.constant 0.000000e+00 : f32
            %396 = memref.load %alloca_2[%327, %331, %336] : memref<2x4x4xf32>
            %397 = arith.addf %cst_134, %396 : f32
            %398 = memref.load %alloca_2[%327, %331, %339] : memref<2x4x4xf32>
            %399 = arith.addf %397, %398 : f32
            %400 = memref.load %alloca_2[%327, %331, %341] : memref<2x4x4xf32>
            %401 = arith.addf %399, %400 : f32
            %402 = memref.load %alloca_2[%327, %331, %343] : memref<2x4x4xf32>
            %403 = arith.addf %401, %402 : f32
            memref.store %403, %alloca_1[%347, %351] : memref<2x4xf32>
            %cst_135 = arith.constant 0.000000e+00 : f32
            %404 = memref.load %alloca_2[%327, %333, %336] : memref<2x4x4xf32>
            %405 = arith.addf %cst_135, %404 : f32
            %406 = memref.load %alloca_2[%327, %333, %339] : memref<2x4x4xf32>
            %407 = arith.addf %405, %406 : f32
            %408 = memref.load %alloca_2[%327, %333, %341] : memref<2x4x4xf32>
            %409 = arith.addf %407, %408 : f32
            %410 = memref.load %alloca_2[%327, %333, %343] : memref<2x4x4xf32>
            %411 = arith.addf %409, %410 : f32
            memref.store %411, %alloca_1[%347, %353] : memref<2x4xf32>
            %cst_136 = arith.constant 0.000000e+00 : f32
            %412 = memref.load %alloca_2[%327, %335, %336] : memref<2x4x4xf32>
            %413 = arith.addf %cst_136, %412 : f32
            %414 = memref.load %alloca_2[%327, %335, %339] : memref<2x4x4xf32>
            %415 = arith.addf %413, %414 : f32
            %416 = memref.load %alloca_2[%327, %335, %341] : memref<2x4x4xf32>
            %417 = arith.addf %415, %416 : f32
            %418 = memref.load %alloca_2[%327, %335, %343] : memref<2x4x4xf32>
            %419 = arith.addf %417, %418 : f32
            memref.store %419, %alloca_1[%347, %355] : memref<2x4xf32>
            %c0_i32_137 = arith.constant 0 : i32
            %420 = arith.index_cast %c0_i32_137 : i32 to index
            %c2_i32_138 = arith.constant 2 : i32
            %421 = arith.index_cast %c2_i32_138 : i32 to index
            %c1_139 = arith.constant 1 : index
            %422 = arith.addi %420, %421 : index
            scf.parallel (%arg6) = (%420) to (%422) step (%c1_139) {
              %c0_i32_172 = arith.constant 0 : i32
              %522 = arith.index_cast %c0_i32_172 : i32 to index
              %c4_i32 = arith.constant 4 : i32
              %523 = arith.index_cast %c4_i32 : i32 to index
              %c1_173 = arith.constant 1 : index
              %524 = arith.addi %522, %523 : index
              scf.parallel (%arg7) = (%522) to (%524) step (%c1_173) {
                %c0_i32_174 = arith.constant 0 : i32
                %525 = arith.index_cast %c0_i32_174 : i32 to index
                %c4_i32_175 = arith.constant 4 : i32
                %526 = arith.index_cast %c4_i32_175 : i32 to index
                %c1_176 = arith.constant 1 : index
                %527 = arith.addi %525, %526 : index
                scf.parallel (%arg8) = (%525) to (%527) step (%c1_176) {
                  %528 = memref.load %alloca_2[%arg6, %arg7, %arg8] : memref<2x4x4xf32>
                  %529 = memref.load %alloca_1[%arg6, %arg7] : memref<2x4xf32>
                  %cst_177 = arith.constant 9.99999997E-7 : f32
                  %530 = arith.addf %529, %cst_177 : f32
                  %531 = arith.divf %528, %530 : f32
                  memref.store %531, %alloca_2[%arg6, %arg7, %arg8] : memref<2x4x4xf32>
                  scf.reduce 
                }
                scf.reduce 
              }
              scf.reduce 
            }
            %c0_i32_140 = arith.constant 0 : i32
            %423 = arith.index_cast %c0_i32_140 : i32 to index
            %c1_i32_141 = arith.constant 1 : i32
            %424 = arith.index_cast %c1_i32_141 : i32 to index
            %c1_142 = arith.constant 1 : index
            %425 = arith.muli %c1_142, %424 : index
            %426 = arith.addi %423, %425 : index
            %c0_i32_143 = arith.constant 0 : i32
            %427 = arith.index_cast %c0_i32_143 : i32 to index
            %c1_i32_144 = arith.constant 1 : i32
            %428 = arith.index_cast %c1_i32_144 : i32 to index
            %c1_145 = arith.constant 1 : index
            %429 = arith.muli %c1_145, %428 : index
            %430 = arith.addi %427, %429 : index
            %c2_146 = arith.constant 2 : index
            %431 = arith.muli %c2_146, %428 : index
            %432 = arith.addi %427, %431 : index
            %c3_147 = arith.constant 3 : index
            %433 = arith.muli %c3_147, %428 : index
            %434 = arith.addi %427, %433 : index
            %c0_i32_148 = arith.constant 0 : i32
            %435 = arith.index_cast %c0_i32_148 : i32 to index
            %c1_i32_149 = arith.constant 1 : i32
            %436 = arith.index_cast %c1_i32_149 : i32 to index
            %c1_150 = arith.constant 1 : index
            %437 = arith.muli %c1_150, %436 : index
            %438 = arith.addi %435, %437 : index
            %c2_151 = arith.constant 2 : index
            %439 = arith.muli %c2_151, %436 : index
            %440 = arith.addi %435, %439 : index
            %c3_152 = arith.constant 3 : index
            %441 = arith.muli %c3_152, %436 : index
            %442 = arith.addi %435, %441 : index
            %c0_i32_153 = arith.constant 0 : i32
            %443 = arith.index_cast %c0_i32_153 : i32 to index
            %c1_i32_154 = arith.constant 1 : i32
            %444 = arith.index_cast %c1_i32_154 : i32 to index
            %c1_155 = arith.constant 1 : index
            %445 = arith.muli %c1_155, %444 : index
            %446 = arith.addi %443, %445 : index
            %c0_i32_156 = arith.constant 0 : i32
            %447 = arith.index_cast %c0_i32_156 : i32 to index
            %c1_i32_157 = arith.constant 1 : i32
            %448 = arith.index_cast %c1_i32_157 : i32 to index
            %c1_158 = arith.constant 1 : index
            %449 = arith.muli %c1_158, %448 : index
            %450 = arith.addi %447, %449 : index
            %c2_159 = arith.constant 2 : index
            %451 = arith.muli %c2_159, %448 : index
            %452 = arith.addi %447, %451 : index
            %c3_160 = arith.constant 3 : index
            %453 = arith.muli %c3_160, %448 : index
            %454 = arith.addi %447, %453 : index
            %cst_161 = arith.constant 0.000000e+00 : f32
            %455 = memref.load %alloca_2[%423, %427, %435] : memref<2x4x4xf32>
            %456 = arith.addf %cst_161, %455 : f32
            %457 = memref.load %alloca_2[%423, %430, %435] : memref<2x4x4xf32>
            %458 = arith.addf %456, %457 : f32
            %459 = memref.load %alloca_2[%423, %432, %435] : memref<2x4x4xf32>
            %460 = arith.addf %458, %459 : f32
            %461 = memref.load %alloca_2[%423, %434, %435] : memref<2x4x4xf32>
            %462 = arith.addf %460, %461 : f32
            memref.store %462, %alloca_0[%443, %447] : memref<2x4xf32>
            %cst_162 = arith.constant 0.000000e+00 : f32
            %463 = memref.load %alloca_2[%423, %427, %438] : memref<2x4x4xf32>
            %464 = arith.addf %cst_162, %463 : f32
            %465 = memref.load %alloca_2[%423, %430, %438] : memref<2x4x4xf32>
            %466 = arith.addf %464, %465 : f32
            %467 = memref.load %alloca_2[%423, %432, %438] : memref<2x4x4xf32>
            %468 = arith.addf %466, %467 : f32
            %469 = memref.load %alloca_2[%423, %434, %438] : memref<2x4x4xf32>
            %470 = arith.addf %468, %469 : f32
            memref.store %470, %alloca_0[%443, %450] : memref<2x4xf32>
            %cst_163 = arith.constant 0.000000e+00 : f32
            %471 = memref.load %alloca_2[%423, %427, %440] : memref<2x4x4xf32>
            %472 = arith.addf %cst_163, %471 : f32
            %473 = memref.load %alloca_2[%423, %430, %440] : memref<2x4x4xf32>
            %474 = arith.addf %472, %473 : f32
            %475 = memref.load %alloca_2[%423, %432, %440] : memref<2x4x4xf32>
            %476 = arith.addf %474, %475 : f32
            %477 = memref.load %alloca_2[%423, %434, %440] : memref<2x4x4xf32>
            %478 = arith.addf %476, %477 : f32
            memref.store %478, %alloca_0[%443, %452] : memref<2x4xf32>
            %cst_164 = arith.constant 0.000000e+00 : f32
            %479 = memref.load %alloca_2[%423, %427, %442] : memref<2x4x4xf32>
            %480 = arith.addf %cst_164, %479 : f32
            %481 = memref.load %alloca_2[%423, %430, %442] : memref<2x4x4xf32>
            %482 = arith.addf %480, %481 : f32
            %483 = memref.load %alloca_2[%423, %432, %442] : memref<2x4x4xf32>
            %484 = arith.addf %482, %483 : f32
            %485 = memref.load %alloca_2[%423, %434, %442] : memref<2x4x4xf32>
            %486 = arith.addf %484, %485 : f32
            memref.store %486, %alloca_0[%443, %454] : memref<2x4xf32>
            %cst_165 = arith.constant 0.000000e+00 : f32
            %487 = memref.load %alloca_2[%426, %427, %435] : memref<2x4x4xf32>
            %488 = arith.addf %cst_165, %487 : f32
            %489 = memref.load %alloca_2[%426, %430, %435] : memref<2x4x4xf32>
            %490 = arith.addf %488, %489 : f32
            %491 = memref.load %alloca_2[%426, %432, %435] : memref<2x4x4xf32>
            %492 = arith.addf %490, %491 : f32
            %493 = memref.load %alloca_2[%426, %434, %435] : memref<2x4x4xf32>
            %494 = arith.addf %492, %493 : f32
            memref.store %494, %alloca_0[%446, %447] : memref<2x4xf32>
            %cst_166 = arith.constant 0.000000e+00 : f32
            %495 = memref.load %alloca_2[%426, %427, %438] : memref<2x4x4xf32>
            %496 = arith.addf %cst_166, %495 : f32
            %497 = memref.load %alloca_2[%426, %430, %438] : memref<2x4x4xf32>
            %498 = arith.addf %496, %497 : f32
            %499 = memref.load %alloca_2[%426, %432, %438] : memref<2x4x4xf32>
            %500 = arith.addf %498, %499 : f32
            %501 = memref.load %alloca_2[%426, %434, %438] : memref<2x4x4xf32>
            %502 = arith.addf %500, %501 : f32
            memref.store %502, %alloca_0[%446, %450] : memref<2x4xf32>
            %cst_167 = arith.constant 0.000000e+00 : f32
            %503 = memref.load %alloca_2[%426, %427, %440] : memref<2x4x4xf32>
            %504 = arith.addf %cst_167, %503 : f32
            %505 = memref.load %alloca_2[%426, %430, %440] : memref<2x4x4xf32>
            %506 = arith.addf %504, %505 : f32
            %507 = memref.load %alloca_2[%426, %432, %440] : memref<2x4x4xf32>
            %508 = arith.addf %506, %507 : f32
            %509 = memref.load %alloca_2[%426, %434, %440] : memref<2x4x4xf32>
            %510 = arith.addf %508, %509 : f32
            memref.store %510, %alloca_0[%446, %452] : memref<2x4xf32>
            %cst_168 = arith.constant 0.000000e+00 : f32
            %511 = memref.load %alloca_2[%426, %427, %442] : memref<2x4x4xf32>
            %512 = arith.addf %cst_168, %511 : f32
            %513 = memref.load %alloca_2[%426, %430, %442] : memref<2x4x4xf32>
            %514 = arith.addf %512, %513 : f32
            %515 = memref.load %alloca_2[%426, %432, %442] : memref<2x4x4xf32>
            %516 = arith.addf %514, %515 : f32
            %517 = memref.load %alloca_2[%426, %434, %442] : memref<2x4x4xf32>
            %518 = arith.addf %516, %517 : f32
            memref.store %518, %alloca_0[%446, %454] : memref<2x4xf32>
            %c0_i32_169 = arith.constant 0 : i32
            %519 = arith.index_cast %c0_i32_169 : i32 to index
            %c2_i32_170 = arith.constant 2 : i32
            %520 = arith.index_cast %c2_i32_170 : i32 to index
            %c1_171 = arith.constant 1 : index
            %521 = arith.addi %519, %520 : index
            scf.parallel (%arg6) = (%519) to (%521) step (%c1_171) {
              %c0_i32_172 = arith.constant 0 : i32
              %522 = arith.index_cast %c0_i32_172 : i32 to index
              %c4_i32 = arith.constant 4 : i32
              %523 = arith.index_cast %c4_i32 : i32 to index
              %c1_173 = arith.constant 1 : index
              %524 = arith.addi %522, %523 : index
              scf.parallel (%arg7) = (%522) to (%524) step (%c1_173) {
                %c0_i32_174 = arith.constant 0 : i32
                %525 = arith.index_cast %c0_i32_174 : i32 to index
                %c4_i32_175 = arith.constant 4 : i32
                %526 = arith.index_cast %c4_i32_175 : i32 to index
                %c1_176 = arith.constant 1 : index
                %527 = arith.addi %525, %526 : index
                scf.parallel (%arg8) = (%525) to (%527) step (%c1_176) {
                  %528 = memref.load %alloca_2[%arg6, %arg7, %arg8] : memref<2x4x4xf32>
                  %529 = memref.load %alloca_0[%arg6, %arg8] : memref<2x4xf32>
                  %cst_177 = arith.constant 9.99999997E-7 : f32
                  %530 = arith.addf %529, %cst_177 : f32
                  %531 = arith.divf %528, %530 : f32
                  memref.store %531, %alloca_2[%arg6, %arg7, %arg8] : memref<2x4x4xf32>
                  scf.reduce 
                }
                scf.reduce 
              }
              scf.reduce 
            }
          }
          %c0_105 = arith.constant 0 : index
          %c2_i32_106 = arith.constant 2 : i32
          %323 = arith.index_cast %c2_i32_106 : i32 to index
          %c1_107 = arith.constant 1 : index
          scf.for %arg5 = %c0_105 to %323 step %c1_107 {
            %c0_108 = arith.constant 0 : index
            %c4_i32 = arith.constant 4 : i32
            %324 = arith.index_cast %c4_i32 : i32 to index
            %c1_109 = arith.constant 1 : index
            scf.for %arg6 = %c0_108 to %324 step %c1_109 {
              %c0_110 = arith.constant 0 : index
              %c4_i32_111 = arith.constant 4 : i32
              %325 = arith.index_cast %c4_i32_111 : i32 to index
              %c1_112 = arith.constant 1 : index
              scf.for %arg7 = %c0_110 to %325 step %c1_112 {
                %326 = arith.index_cast %arg2 : index to i32
                %c2_i32_113 = arith.constant 2 : i32
                %327 = arith.muli %326, %c2_i32_113 : i32
                %328 = arith.index_cast %327 : i32 to index
                %329 = arith.addi %arg5, %328 : index
                %true = arith.constant true
                %c0_114 = arith.constant 0 : index
                %330 = arith.cmpi sge, %329, %c0_114 : index
                %331 = arith.andi %true, %330 : i1
                %332 = arith.cmpi slt, %329, %dim : index
                %333 = arith.andi %331, %332 : i1
                %c4_i32_115 = arith.constant 4 : i32
                %334 = arith.index_cast %c4_i32_115 : i32 to index
                %c0_116 = arith.constant 0 : index
                %335 = arith.cmpi sge, %arg6, %c0_116 : index
                %336 = arith.andi %333, %335 : i1
                %337 = arith.cmpi slt, %arg6, %334 : index
                %338 = arith.andi %336, %337 : i1
                %c4_i32_117 = arith.constant 4 : i32
                %339 = arith.index_cast %c4_i32_117 : i32 to index
                %c0_118 = arith.constant 0 : index
                %340 = arith.cmpi sge, %arg7, %c0_118 : index
                %341 = arith.andi %338, %340 : i1
                %342 = arith.cmpi slt, %arg7, %339 : index
                %343 = arith.andi %341, %342 : i1
                scf.if %343 {
                  %true_119 = arith.constant true
                  %c2_i32_120 = arith.constant 2 : i32
                  %344 = arith.index_cast %c2_i32_120 : i32 to index
                  %c0_121 = arith.constant 0 : index
                  %345 = arith.cmpi sge, %arg5, %c0_121 : index
                  %346 = arith.andi %true_119, %345 : i1
                  %347 = arith.cmpi slt, %arg5, %344 : index
                  %348 = arith.andi %346, %347 : i1
                  %c4_i32_122 = arith.constant 4 : i32
                  %349 = arith.index_cast %c4_i32_122 : i32 to index
                  %c0_123 = arith.constant 0 : index
                  %350 = arith.cmpi sge, %arg6, %c0_123 : index
                  %351 = arith.andi %348, %350 : i1
                  %352 = arith.cmpi slt, %arg6, %349 : index
                  %353 = arith.andi %351, %352 : i1
                  %c4_i32_124 = arith.constant 4 : i32
                  %354 = arith.index_cast %c4_i32_124 : i32 to index
                  %c0_125 = arith.constant 0 : index
                  %355 = arith.cmpi sge, %arg7, %c0_125 : index
                  %356 = arith.andi %353, %355 : i1
                  %357 = arith.cmpi slt, %arg7, %354 : index
                  %358 = arith.andi %356, %357 : i1
                  %359 = scf.if %358 -> (f32) {
                    %360 = memref.load %alloca_2[%arg5, %arg6, %arg7] : memref<2x4x4xf32>
                    scf.yield %360 : f32
                  } else {
                    %cst_126 = arith.constant 0.000000e+00 : f32
                    scf.yield %cst_126 : f32
                  }
                  memref.store %359, %arg1[%329, %arg6, %arg7] : memref<?x4x4xf32>
                }
              }
            }
          }
        }
      }
    }
    return
  }
}

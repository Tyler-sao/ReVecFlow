#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
#map3 = affine_map<(d0, d1) -> (d0, 0)>
#map4 = affine_map<(d0, d1) -> (d0, 0)>
#SPARSE = #sparse_tensor.encoding<{
  map = (i,j) -> (i : compressed, j : compressed)
}>
#SPARSE1D = #sparse_tensor.encoding<{
  map = (i) -> (i : dense)
}>
// GIN (Xu et al., 2019), GIN-0, 2 layers, hidden 64, fully fused
module attributes {torch.debug_module_name = "GIN"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<1767x50xf32>, %arg1: tensor<1767x1767xf32, #SPARSE>, %arg2: tensor<50x64xf32, #SPARSE>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64xf32>, %arg7: tensor<64xf32>, %arg8: tensor<64x64xf32, #SPARSE>, %arg9: tensor<64xf32>, %arg10: tensor<64xf32>, %arg11: tensor<64xf32>, %arg12: tensor<64xf32>, %arg13: tensor<64xf32>, %arg14: tensor<64x64xf32, #SPARSE>, %arg15: tensor<64xf32>, %arg16: tensor<64xf32>, %arg17: tensor<64xf32>, %arg18: tensor<64xf32>, %arg19: tensor<64xf32>, %arg20: tensor<64x121xf32, #SPARSE>, %arg21: tensor<121xf32>) -> tensor<1767x121xf32> {
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_1 = arith.constant 0xFF800000 : f32
    %cst_eps = arith.constant 9.99999974E-6 : f32
    %cst_alpha = arith.constant 1.000000e-01 : f32
    %cst_keep = arith.constant 9.000000e-01 : f32
    %v1 = tensor.empty() : tensor<1767x50xf32>
    %v2 = linalg.fill ins(%cst_0 : f32) outs(%v1 : tensor<1767x50xf32>) -> tensor<1767x50xf32>
    %v3 = linalg.matmul ins(%arg1, %arg0 : tensor<1767x1767xf32, #SPARSE>, tensor<1767x50xf32>) outs(%v2 : tensor<1767x50xf32>) -> tensor<1767x50xf32>
    %v4 = tensor.empty() : tensor<1767x50xf32>
    %v5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%v3, %arg0 : tensor<1767x50xf32>, tensor<1767x50xf32>) outs(%v4 : tensor<1767x50xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.addf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x50xf32>
    %v6 = tensor.empty() : tensor<1767x64xf32>
    %v7 = linalg.fill ins(%cst_0 : f32) outs(%v6 : tensor<1767x64xf32>) -> tensor<1767x64xf32>
    %v8 = linalg.matmul ins(%v5, %arg2 : tensor<1767x50xf32>, tensor<50x64xf32, #SPARSE>) outs(%v7 : tensor<1767x64xf32>) -> tensor<1767x64xf32>
    %v9 = tensor.empty() : tensor<1767x64xf32>
    %v10 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%v8, %arg3 : tensor<1767x64xf32>, tensor<64xf32>) outs(%v9 : tensor<1767x64xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.addf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x64xf32>
    %v11 = tensor.empty() : tensor<1767x64xf32>
    %v12 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%v10, %arg4 : tensor<1767x64xf32>, tensor<64xf32>) outs(%v11 : tensor<1767x64xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.subf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x64xf32>
    %v13 = tensor.empty() : tensor<1767x64xf32>
    %v14 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%v12, %arg5 : tensor<1767x64xf32>, tensor<64xf32>) outs(%v13 : tensor<1767x64xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.mulf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x64xf32>
    %v15 = tensor.empty() : tensor<1767x64xf32>
    %v16 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%v14, %arg6 : tensor<1767x64xf32>, tensor<64xf32>) outs(%v15 : tensor<1767x64xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.mulf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x64xf32>
    %v17 = tensor.empty() : tensor<1767x64xf32>
    %v18 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%v16, %arg7 : tensor<1767x64xf32>, tensor<64xf32>) outs(%v17 : tensor<1767x64xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.addf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x64xf32>
    %v19 = tensor.empty() : tensor<1767x64xf32>
    %v20 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%v18 : tensor<1767x64xf32>) outs(%v19 : tensor<1767x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %29 = arith.cmpf ugt, %in, %cst_0 : f32
      %30 = arith.select %29, %in, %cst_0 : f32
      linalg.yield %30 : f32
    } -> tensor<1767x64xf32>
    %v21 = tensor.empty() : tensor<1767x64xf32>
    %v22 = linalg.fill ins(%cst_0 : f32) outs(%v21 : tensor<1767x64xf32>) -> tensor<1767x64xf32>
    %v23 = linalg.matmul ins(%v20, %arg8 : tensor<1767x64xf32>, tensor<64x64xf32, #SPARSE>) outs(%v22 : tensor<1767x64xf32>) -> tensor<1767x64xf32>
    %v24 = tensor.empty() : tensor<1767x64xf32>
    %v25 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%v23, %arg9 : tensor<1767x64xf32>, tensor<64xf32>) outs(%v24 : tensor<1767x64xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.addf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x64xf32>
    %v26 = tensor.empty() : tensor<1767x64xf32>
    %v27 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%v25, %arg10 : tensor<1767x64xf32>, tensor<64xf32>) outs(%v26 : tensor<1767x64xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.subf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x64xf32>
    %v28 = tensor.empty() : tensor<1767x64xf32>
    %v29 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%v27, %arg11 : tensor<1767x64xf32>, tensor<64xf32>) outs(%v28 : tensor<1767x64xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.mulf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x64xf32>
    %v30 = tensor.empty() : tensor<1767x64xf32>
    %v31 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%v29, %arg12 : tensor<1767x64xf32>, tensor<64xf32>) outs(%v30 : tensor<1767x64xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.mulf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x64xf32>
    %v32 = tensor.empty() : tensor<1767x64xf32>
    %v33 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%v31, %arg13 : tensor<1767x64xf32>, tensor<64xf32>) outs(%v32 : tensor<1767x64xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.addf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x64xf32>
    %v34 = tensor.empty() : tensor<1767x64xf32>
    %v35 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%v33 : tensor<1767x64xf32>) outs(%v34 : tensor<1767x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %29 = arith.cmpf ugt, %in, %cst_0 : f32
      %30 = arith.select %29, %in, %cst_0 : f32
      linalg.yield %30 : f32
    } -> tensor<1767x64xf32>
    %v36 = tensor.empty() : tensor<1767x64xf32>
    %v37 = linalg.fill ins(%cst_0 : f32) outs(%v36 : tensor<1767x64xf32>) -> tensor<1767x64xf32>
    %v38 = linalg.matmul ins(%arg1, %v35 : tensor<1767x1767xf32, #SPARSE>, tensor<1767x64xf32>) outs(%v37 : tensor<1767x64xf32>) -> tensor<1767x64xf32>
    %v39 = tensor.empty() : tensor<1767x64xf32>
    %v40 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%v38, %v35 : tensor<1767x64xf32>, tensor<1767x64xf32>) outs(%v39 : tensor<1767x64xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.addf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x64xf32>
    %v41 = tensor.empty() : tensor<1767x64xf32>
    %v42 = linalg.fill ins(%cst_0 : f32) outs(%v41 : tensor<1767x64xf32>) -> tensor<1767x64xf32>
    %v43 = linalg.matmul ins(%v40, %arg14 : tensor<1767x64xf32>, tensor<64x64xf32, #SPARSE>) outs(%v42 : tensor<1767x64xf32>) -> tensor<1767x64xf32>
    %v44 = tensor.empty() : tensor<1767x64xf32>
    %v45 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%v43, %arg15 : tensor<1767x64xf32>, tensor<64xf32>) outs(%v44 : tensor<1767x64xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.addf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x64xf32>
    %v46 = tensor.empty() : tensor<1767x64xf32>
    %v47 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%v45, %arg16 : tensor<1767x64xf32>, tensor<64xf32>) outs(%v46 : tensor<1767x64xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.subf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x64xf32>
    %v48 = tensor.empty() : tensor<1767x64xf32>
    %v49 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%v47, %arg17 : tensor<1767x64xf32>, tensor<64xf32>) outs(%v48 : tensor<1767x64xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.mulf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x64xf32>
    %v50 = tensor.empty() : tensor<1767x64xf32>
    %v51 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%v49, %arg18 : tensor<1767x64xf32>, tensor<64xf32>) outs(%v50 : tensor<1767x64xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.mulf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x64xf32>
    %v52 = tensor.empty() : tensor<1767x64xf32>
    %v53 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%v51, %arg19 : tensor<1767x64xf32>, tensor<64xf32>) outs(%v52 : tensor<1767x64xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.addf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x64xf32>
    %v54 = tensor.empty() : tensor<1767x64xf32>
    %v55 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%v53 : tensor<1767x64xf32>) outs(%v54 : tensor<1767x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %29 = arith.cmpf ugt, %in, %cst_0 : f32
      %30 = arith.select %29, %in, %cst_0 : f32
      linalg.yield %30 : f32
    } -> tensor<1767x64xf32>
    %v56 = tensor.empty() : tensor<1767x121xf32>
    %v57 = linalg.fill ins(%cst_0 : f32) outs(%v56 : tensor<1767x121xf32>) -> tensor<1767x121xf32>
    %v58 = linalg.matmul ins(%v55, %arg20 : tensor<1767x64xf32>, tensor<64x121xf32, #SPARSE>) outs(%v57 : tensor<1767x121xf32>) -> tensor<1767x121xf32>
    %v59 = tensor.empty() : tensor<1767x121xf32>
    %v60 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%v58, %arg21 : tensor<1767x121xf32>, tensor<121xf32>) outs(%v59 : tensor<1767x121xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.addf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x121xf32>
    %v73 = tensor.empty() : tensor<1767x121xf32>
    %v61 = tensor.empty() : tensor<1767x1xi64>
    %v62 = linalg.fill ins(%c0_i64 : i64) outs(%v61 : tensor<1767x1xi64>) -> tensor<1767x1xi64>
    %v63 = tensor.empty() : tensor<1767x1xf32>
    %v64 = linalg.fill ins(%cst_1 : f32) outs(%v63 : tensor<1767x1xf32>) -> tensor<1767x1xf32>
    %v65:2 = linalg.generic {indexing_maps = [#map, #map3, #map3], iterator_types = ["parallel", "reduction"]} ins(%v60 : tensor<1767x121xf32>) outs(%v64, %v62 : tensor<1767x1xf32>, tensor<1767x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_5: i64):
      %29 = linalg.index 1 : index
      %30 = arith.index_cast %29 : index to i64
      %31 = arith.maximumf %in, %out : f32
      %32 = arith.cmpf ogt, %in, %out : f32
      %33 = arith.select %32, %30, %out_5 : i64
      linalg.yield %31, %33 : f32, i64
    } -> (tensor<1767x1xf32>, tensor<1767x1xi64>)
    %v66 = linalg.generic {indexing_maps = [#map, #map4, #map], iterator_types = ["parallel", "parallel"]} ins(%v60, %v65#0 : tensor<1767x121xf32>, tensor<1767x1xf32>) outs(%v73 : tensor<1767x121xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.subf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x121xf32>
    %v67 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%v66 : tensor<1767x121xf32>) outs(%v73 : tensor<1767x121xf32>) {
    ^bb0(%in: f32, %out: f32):
      %29 = math.exp %in : f32
      linalg.yield %29 : f32
    } -> tensor<1767x121xf32>
    %v68 = tensor.empty() : tensor<1767x1xf32>
    %v69 = linalg.fill ins(%cst_0 : f32) outs(%v68 : tensor<1767x1xf32>) -> tensor<1767x1xf32>
    %v70 = linalg.generic {indexing_maps = [#map, #map4], iterator_types = ["parallel", "reduction"]} ins(%v67 : tensor<1767x121xf32>) outs(%v69 : tensor<1767x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %29 = arith.addf %in, %out : f32
      linalg.yield %29 : f32
    } -> tensor<1767x1xf32>
    %v71 = linalg.generic {indexing_maps = [#map4, #map], iterator_types = ["parallel", "parallel"]} ins(%v70 : tensor<1767x1xf32>) outs(%v68 : tensor<1767x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %29 = math.log %in : f32
      linalg.yield %29 : f32
    } -> tensor<1767x1xf32>
    %v72 = linalg.generic {indexing_maps = [#map, #map4, #map], iterator_types = ["parallel", "parallel"]} ins(%v67, %v70 : tensor<1767x121xf32>, tensor<1767x1xf32>) outs(%v73 : tensor<1767x121xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.divf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x121xf32>
    return %v72 : tensor<1767x121xf32>
  }
}

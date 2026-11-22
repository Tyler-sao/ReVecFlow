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
// DIAGNOSTIC, check only: GIN layer 2 without the self term
module attributes {torch.debug_module_name = "GIN"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<1767x64xf32>, %arg1: tensor<1767x1767xf32, #SPARSE>, %arg2: tensor<64x64xf32, #SPARSE>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64xf32>, %arg7: tensor<64xf32>, %arg8: tensor<64x121xf32, #SPARSE>, %arg9: tensor<121xf32>) -> tensor<1767x121xf32> {
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_1 = arith.constant 0xFF800000 : f32
    %cst_eps = arith.constant 9.99999974E-6 : f32
    %cst_alpha = arith.constant 1.000000e-01 : f32
    %cst_keep = arith.constant 9.000000e-01 : f32
    %v1 = tensor.empty() : tensor<1767x64xf32>
    %v2 = linalg.fill ins(%cst_0 : f32) outs(%v1 : tensor<1767x64xf32>) -> tensor<1767x64xf32>
    %v3 = linalg.matmul ins(%arg1, %arg0 : tensor<1767x1767xf32, #SPARSE>, tensor<1767x64xf32>) outs(%v2 : tensor<1767x64xf32>) -> tensor<1767x64xf32>
    %v4 = tensor.empty() : tensor<1767x64xf32>
    %v5 = linalg.fill ins(%cst_0 : f32) outs(%v4 : tensor<1767x64xf32>) -> tensor<1767x64xf32>
    %v6 = linalg.matmul ins(%v3, %arg2 : tensor<1767x64xf32>, tensor<64x64xf32, #SPARSE>) outs(%v5 : tensor<1767x64xf32>) -> tensor<1767x64xf32>
    %v7 = tensor.empty() : tensor<1767x64xf32>
    %v8 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%v6, %arg3 : tensor<1767x64xf32>, tensor<64xf32>) outs(%v7 : tensor<1767x64xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.addf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x64xf32>
    %v9 = tensor.empty() : tensor<1767x64xf32>
    %v10 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%v8, %arg4 : tensor<1767x64xf32>, tensor<64xf32>) outs(%v9 : tensor<1767x64xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.subf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x64xf32>
    %v11 = tensor.empty() : tensor<1767x64xf32>
    %v12 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%v10, %arg5 : tensor<1767x64xf32>, tensor<64xf32>) outs(%v11 : tensor<1767x64xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.mulf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x64xf32>
    %v13 = tensor.empty() : tensor<1767x64xf32>
    %v14 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%v12, %arg6 : tensor<1767x64xf32>, tensor<64xf32>) outs(%v13 : tensor<1767x64xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.mulf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x64xf32>
    %v15 = tensor.empty() : tensor<1767x64xf32>
    %v16 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%v14, %arg7 : tensor<1767x64xf32>, tensor<64xf32>) outs(%v15 : tensor<1767x64xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.addf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x64xf32>
    %v17 = tensor.empty() : tensor<1767x64xf32>
    %v18 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%v16 : tensor<1767x64xf32>) outs(%v17 : tensor<1767x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %29 = arith.cmpf ugt, %in, %cst_0 : f32
      %30 = arith.select %29, %in, %cst_0 : f32
      linalg.yield %30 : f32
    } -> tensor<1767x64xf32>
    %v19 = tensor.empty() : tensor<1767x121xf32>
    %v20 = linalg.fill ins(%cst_0 : f32) outs(%v19 : tensor<1767x121xf32>) -> tensor<1767x121xf32>
    %v21 = linalg.matmul ins(%v18, %arg8 : tensor<1767x64xf32>, tensor<64x121xf32, #SPARSE>) outs(%v20 : tensor<1767x121xf32>) -> tensor<1767x121xf32>
    %v22 = tensor.empty() : tensor<1767x121xf32>
    %v23 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%v21, %arg9 : tensor<1767x121xf32>, tensor<121xf32>) outs(%v22 : tensor<1767x121xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.addf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x121xf32>
    return %v23 : tensor<1767x121xf32>
  }
}

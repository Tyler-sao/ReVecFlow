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
// CHECK-ONLY (values before the softmax). GIN partially fused, layer 2 + softmax; input = synthetic hidden activation
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
    %v5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%v3, %arg0 : tensor<1767x64xf32>, tensor<1767x64xf32>) outs(%v4 : tensor<1767x64xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.addf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x64xf32>
    %v6 = tensor.empty() : tensor<1767x64xf32>
    %v7 = linalg.fill ins(%cst_0 : f32) outs(%v6 : tensor<1767x64xf32>) -> tensor<1767x64xf32>
    %v8 = linalg.matmul ins(%v5, %arg2 : tensor<1767x64xf32>, tensor<64x64xf32, #SPARSE>) outs(%v7 : tensor<1767x64xf32>) -> tensor<1767x64xf32>
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
    %v21 = tensor.empty() : tensor<1767x121xf32>
    %v22 = linalg.fill ins(%cst_0 : f32) outs(%v21 : tensor<1767x121xf32>) -> tensor<1767x121xf32>
    %v23 = linalg.matmul ins(%v20, %arg8 : tensor<1767x64xf32>, tensor<64x121xf32, #SPARSE>) outs(%v22 : tensor<1767x121xf32>) -> tensor<1767x121xf32>
    %v24 = tensor.empty() : tensor<1767x121xf32>
    %v25 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%v23, %arg9 : tensor<1767x121xf32>, tensor<121xf32>) outs(%v24 : tensor<1767x121xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.addf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x121xf32>
    return %v25 : tensor<1767x121xf32>
  }
}

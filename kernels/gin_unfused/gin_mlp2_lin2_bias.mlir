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

// GIN unfused, layer 2 MLP linear 2: bias
func.func @matmul1 (%arg0: tensor<1767x121xf32>, %arg1: tensor<121xf32>, %argout: tensor<1767x121xf32>) -> tensor<1767x121xf32> {
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_1 = arith.constant 0xFF800000 : f32
    %cst_eps = arith.constant 9.99999974E-6 : f32
    %cst_alpha = arith.constant 1.000000e-01 : f32
    %cst_keep = arith.constant 9.000000e-01 : f32
    %v1 = tensor.empty() : tensor<1767x121xf32>
    %v2 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1767x121xf32>, tensor<121xf32>) outs(%v1 : tensor<1767x121xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.addf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x121xf32>
    return %v2 : tensor<1767x121xf32>
}

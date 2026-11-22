#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d1)>
#map3 = affine_map<(d0, d1) -> (d0, 0)>

#SPARSE = #sparse_tensor.encoding<{
  map = (i,j) -> (i : compressed, j : compressed)
}>
#SPARSE1D = #sparse_tensor.encoding<{
  map = (i) -> (i : compressed)
}>

// SGC unfused: classifier bias (as gcn_linear1_bias)
func.func @matmul1 (%arg0 : tensor<1767x121xf32>, %arg1 : tensor<121xf32>, %argout : tensor<1767x121xf32>) -> tensor<1767x121xf32> {
  %2 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1767x121xf32>, tensor<121xf32>) outs(%argout : tensor<1767x121xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.addf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x121xf32>
  return %2 : tensor<1767x121xf32>
}

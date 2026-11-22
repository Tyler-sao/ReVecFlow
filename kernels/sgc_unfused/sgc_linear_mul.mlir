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

// SGC unfused: classifier product P . W (as gcn_linear1_mul)
func.func @matmul1 (%arg0 : tensor<1767x50xf32>, %arg1 : tensor<50x121xf32, #SPARSE>, %argout : tensor<1767x121xf32>) -> tensor<1767x121xf32> {
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<1767x50xf32>, tensor<50x121xf32, #SPARSE>) outs(%argout : tensor<1767x121xf32>) -> tensor<1767x121xf32>
  return %2 : tensor<1767x121xf32>
}

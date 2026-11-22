// V3 H_G KERNEL (new file, 2026-10-19): sgc_unfused/sgc_prop1.mlir with the dense operand N x F (1767x50) changed to N x C (1767x121); nothing else changed
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

// SGC unfused: propagation 1, S . P (one SpMM, as gcn_adj_x1)
func.func @matmul1 (%arg0 : tensor<1767x1767xf32, #SPARSE>, %arg1 : tensor<1767x121xf32>, %argout : tensor<1767x121xf32>) -> tensor<1767x121xf32> {
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<1767x1767xf32, #SPARSE>, tensor<1767x121xf32>) outs(%argout : tensor<1767x121xf32>) -> tensor<1767x121xf32>
  return %2 : tensor<1767x121xf32>
}

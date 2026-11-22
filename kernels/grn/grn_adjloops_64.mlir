// GRANII-dagger KERNEL (new file, 2026-10-27): gin_adj_h2.mlir unchanged, data A + I
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

// GIN unfused, layer 2: A . H (as gcn_adj_x1)
func.func @matmul1 (%arg0: tensor<1767x1767xf32, #SPARSE>, %arg1: tensor<1767x64xf32>, %argout: tensor<1767x64xf32>) -> tensor<1767x64xf32> {
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_1 = arith.constant 0xFF800000 : f32
    %cst_eps = arith.constant 9.99999974E-6 : f32
    %cst_alpha = arith.constant 1.000000e-01 : f32
    %cst_keep = arith.constant 9.000000e-01 : f32
    %v1 = tensor.empty() : tensor<1767x64xf32>
    %v2 = linalg.fill ins(%cst_0 : f32) outs(%v1 : tensor<1767x64xf32>) -> tensor<1767x64xf32>
    %v3 = linalg.matmul ins(%arg0, %arg1 : tensor<1767x1767xf32, #SPARSE>, tensor<1767x64xf32>) outs(%v2 : tensor<1767x64xf32>) -> tensor<1767x64xf32>
    return %v3 : tensor<1767x64xf32>
}

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
// SGC partially fused, stage 1 (propagation): S . (S . X)
module attributes {torch.debug_module_name = "SGC"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<1767x50xf32>, %arg1: tensor<1767x1767xf32, #SPARSE>) -> tensor<1767x50xf32> {
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1767x50xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1767x50xf32>) -> tensor<1767x50xf32>
    %2 = linalg.matmul ins(%arg1, %arg0 : tensor<1767x1767xf32, #SPARSE>, tensor<1767x50xf32>) outs(%1 : tensor<1767x50xf32>) -> tensor<1767x50xf32>
    %3 = linalg.matmul ins(%arg1, %2 : tensor<1767x1767xf32, #SPARSE>, tensor<1767x50xf32>) outs(%1 : tensor<1767x50xf32>) -> tensor<1767x50xf32>
    return %3 : tensor<1767x50xf32>
  }
}

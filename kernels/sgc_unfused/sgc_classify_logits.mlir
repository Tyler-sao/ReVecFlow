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
// CHECK-ONLY (values before the softmax, not part of any configuration). SGC partially fused, stage 2 (classifier): softmax(P . W + b)
module attributes {torch.debug_module_name = "SGC"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<1767x50xf32>) -> tensor<1767x121xf32> {
    %cst = arith.constant dense_resource<__elided__> : tensor<121x50xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_1 = arith.constant 0xFF800000 : f32
    %cst_4 = arith.constant dense_resource<__elided__> : tensor<121xf32>
    %11 = tensor.empty() : tensor<50x121xf32, #SPARSE>
    %12 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst : tensor<121x50xf32>) outs(%11 : tensor<50x121xf32, #SPARSE>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<50x121xf32, #SPARSE>
    %13 = tensor.empty() : tensor<1767x121xf32>
    %14 = linalg.fill ins(%cst_0 : f32) outs(%13 : tensor<1767x121xf32>) -> tensor<1767x121xf32>
    %15 = linalg.matmul ins(%arg0, %12 : tensor<1767x50xf32>, tensor<50x121xf32, #SPARSE>) outs(%14 : tensor<1767x121xf32>) -> tensor<1767x121xf32>
    %16 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%15, %cst_4 : tensor<1767x121xf32>, tensor<121xf32>) outs(%13 : tensor<1767x121xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.addf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x121xf32>
    return %16 : tensor<1767x121xf32>
  }
}

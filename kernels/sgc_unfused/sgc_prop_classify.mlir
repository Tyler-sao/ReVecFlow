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
// SGC partially fused: S . P . W + b, softmax (second propagation with the classifier)
module attributes {torch.debug_module_name = "SGC"} {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @forward(%arg0: tensor<1767x1767xf32, #SPARSE>, %arg1: tensor<1767x50xf32>, %arg2: tensor<50x121xf32, #SPARSE>, %arg3: tensor<121xf32>) -> tensor<1767x121xf32> {
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_1 = arith.constant 0xFF800000 : f32
    %cst_eps = arith.constant 9.99999974E-6 : f32
    %cst_alpha = arith.constant 1.000000e-01 : f32
    %cst_keep = arith.constant 9.000000e-01 : f32
    %v1 = tensor.empty() : tensor<1767x50xf32>
    %v2 = linalg.fill ins(%cst_0 : f32) outs(%v1 : tensor<1767x50xf32>) -> tensor<1767x50xf32>
    %v3 = linalg.matmul ins(%arg0, %arg1 : tensor<1767x1767xf32, #SPARSE>, tensor<1767x50xf32>) outs(%v2 : tensor<1767x50xf32>) -> tensor<1767x50xf32>
    %v4 = tensor.empty() : tensor<1767x121xf32>
    %v5 = linalg.fill ins(%cst_0 : f32) outs(%v4 : tensor<1767x121xf32>) -> tensor<1767x121xf32>
    %v6 = linalg.matmul ins(%v3, %arg2 : tensor<1767x50xf32>, tensor<50x121xf32, #SPARSE>) outs(%v5 : tensor<1767x121xf32>) -> tensor<1767x121xf32>
    %v7 = tensor.empty() : tensor<1767x121xf32>
    %v8 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%v6, %arg3 : tensor<1767x121xf32>, tensor<121xf32>) outs(%v7 : tensor<1767x121xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.addf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x121xf32>
    %v21 = tensor.empty() : tensor<1767x121xf32>
    %v9 = tensor.empty() : tensor<1767x1xi64>
    %v10 = linalg.fill ins(%c0_i64 : i64) outs(%v9 : tensor<1767x1xi64>) -> tensor<1767x1xi64>
    %v11 = tensor.empty() : tensor<1767x1xf32>
    %v12 = linalg.fill ins(%cst_1 : f32) outs(%v11 : tensor<1767x1xf32>) -> tensor<1767x1xf32>
    %v13:2 = linalg.generic {indexing_maps = [#map, #map3, #map3], iterator_types = ["parallel", "reduction"]} ins(%v8 : tensor<1767x121xf32>) outs(%v12, %v10 : tensor<1767x1xf32>, tensor<1767x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_5: i64):
      %29 = linalg.index 1 : index
      %30 = arith.index_cast %29 : index to i64
      %31 = arith.maximumf %in, %out : f32
      %32 = arith.cmpf ogt, %in, %out : f32
      %33 = arith.select %32, %30, %out_5 : i64
      linalg.yield %31, %33 : f32, i64
    } -> (tensor<1767x1xf32>, tensor<1767x1xi64>)
    %v14 = linalg.generic {indexing_maps = [#map, #map4, #map], iterator_types = ["parallel", "parallel"]} ins(%v8, %v13#0 : tensor<1767x121xf32>, tensor<1767x1xf32>) outs(%v21 : tensor<1767x121xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.subf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x121xf32>
    %v15 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%v14 : tensor<1767x121xf32>) outs(%v21 : tensor<1767x121xf32>) {
    ^bb0(%in: f32, %out: f32):
      %29 = math.exp %in : f32
      linalg.yield %29 : f32
    } -> tensor<1767x121xf32>
    %v16 = tensor.empty() : tensor<1767x1xf32>
    %v17 = linalg.fill ins(%cst_0 : f32) outs(%v16 : tensor<1767x1xf32>) -> tensor<1767x1xf32>
    %v18 = linalg.generic {indexing_maps = [#map, #map4], iterator_types = ["parallel", "reduction"]} ins(%v15 : tensor<1767x121xf32>) outs(%v17 : tensor<1767x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %29 = arith.addf %in, %out : f32
      linalg.yield %29 : f32
    } -> tensor<1767x1xf32>
    %v19 = linalg.generic {indexing_maps = [#map4, #map], iterator_types = ["parallel", "parallel"]} ins(%v18 : tensor<1767x1xf32>) outs(%v16 : tensor<1767x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %29 = math.log %in : f32
      linalg.yield %29 : f32
    } -> tensor<1767x1xf32>
    %v20 = linalg.generic {indexing_maps = [#map, #map4, #map], iterator_types = ["parallel", "parallel"]} ins(%v15, %v18 : tensor<1767x121xf32>, tensor<1767x1xf32>) outs(%v21 : tensor<1767x121xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.divf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x121xf32>
    return %v20 : tensor<1767x121xf32>
  }
}

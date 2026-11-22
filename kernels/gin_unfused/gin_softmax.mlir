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

// GIN unfused, softmax over classes (as gcn_softmax)
func.func @matmul1 (%arg0: tensor<1767x121xf32>, %argout: tensor<1767x121xf32>) -> tensor<1767x121xf32> {
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_1 = arith.constant 0xFF800000 : f32
    %cst_eps = arith.constant 9.99999974E-6 : f32
    %cst_alpha = arith.constant 1.000000e-01 : f32
    %cst_keep = arith.constant 9.000000e-01 : f32
    %v13 = tensor.empty() : tensor<1767x121xf32>
    %v1 = tensor.empty() : tensor<1767x1xi64>
    %v2 = linalg.fill ins(%c0_i64 : i64) outs(%v1 : tensor<1767x1xi64>) -> tensor<1767x1xi64>
    %v3 = tensor.empty() : tensor<1767x1xf32>
    %v4 = linalg.fill ins(%cst_1 : f32) outs(%v3 : tensor<1767x1xf32>) -> tensor<1767x1xf32>
    %v5:2 = linalg.generic {indexing_maps = [#map, #map3, #map3], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<1767x121xf32>) outs(%v4, %v2 : tensor<1767x1xf32>, tensor<1767x1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_5: i64):
      %29 = linalg.index 1 : index
      %30 = arith.index_cast %29 : index to i64
      %31 = arith.maximumf %in, %out : f32
      %32 = arith.cmpf ogt, %in, %out : f32
      %33 = arith.select %32, %30, %out_5 : i64
      linalg.yield %31, %33 : f32, i64
    } -> (tensor<1767x1xf32>, tensor<1767x1xi64>)
    %v6 = linalg.generic {indexing_maps = [#map, #map4, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %v5#0 : tensor<1767x121xf32>, tensor<1767x1xf32>) outs(%v13 : tensor<1767x121xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.subf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x121xf32>
    %v7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%v6 : tensor<1767x121xf32>) outs(%v13 : tensor<1767x121xf32>) {
    ^bb0(%in: f32, %out: f32):
      %29 = math.exp %in : f32
      linalg.yield %29 : f32
    } -> tensor<1767x121xf32>
    %v8 = tensor.empty() : tensor<1767x1xf32>
    %v9 = linalg.fill ins(%cst_0 : f32) outs(%v8 : tensor<1767x1xf32>) -> tensor<1767x1xf32>
    %v10 = linalg.generic {indexing_maps = [#map, #map4], iterator_types = ["parallel", "reduction"]} ins(%v7 : tensor<1767x121xf32>) outs(%v9 : tensor<1767x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %29 = arith.addf %in, %out : f32
      linalg.yield %29 : f32
    } -> tensor<1767x1xf32>
    %v11 = linalg.generic {indexing_maps = [#map4, #map], iterator_types = ["parallel", "parallel"]} ins(%v10 : tensor<1767x1xf32>) outs(%v8 : tensor<1767x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %29 = math.log %in : f32
      linalg.yield %29 : f32
    } -> tensor<1767x1xf32>
    %v12 = linalg.generic {indexing_maps = [#map, #map4, #map], iterator_types = ["parallel", "parallel"]} ins(%v7, %v10 : tensor<1767x121xf32>, tensor<1767x1xf32>) outs(%v13 : tensor<1767x121xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %29 = arith.divf %in, %in_5 : f32
      linalg.yield %29 : f32
    } -> tensor<1767x121xf32>
    return %v12 : tensor<1767x121xf32>
}

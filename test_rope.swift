import MLX
import MLXNN
let r = RoPE(dimensions: 64)
let x = MLXArray.ones([2, 4, 64]) // [Batch, SeqLen, Dim]
let y = r(x)
print(y.shape)

/// Phase 0 feasibility probe — validates that mlx-swift provides every op
/// needed by the ACE-Step v1.5 DiT + LM + VAE before the full port begins.
///
/// Test groups map directly to architecture components:
///   1. Core tensor ops         — all DiT operations
///   2. Linear                  — all projection layers
///   3. Normalization            — DiT blocks, LM blocks
///   4. Embedding               — token embedding, lyric encoder
///   5. MultiHeadAttention      — self-attention, cross-attention
///   6. RoPE                    — DiT and LM position encoding
///   7. Conv1d                  — AudioVAE encoder/decoder
///   8. DDIM step (raw ops)     — TurboSampler (no built-in sampler in mlx-swift)
///   9. Safetensors loading     — DiTWeightLoader, LMWeightLoader
///
/// Exit criterion: all tests pass → proceed to Phase 1.
/// Any XCTFail → document the gap and its workaround before continuing.

import XCTest
import MLX
import MLXNN
import MLXRandom

final class FeasibilityProbeTests: XCTestCase {

    // MARK: - 1. Core Tensor Operations

    func testTensorCreation() {
        let z = MLXArray.zeros([2, 4])
        XCTAssertEqual(z.shape, [2, 4])
        XCTAssertEqual(z.dtype, .float32)

        let o = MLXArray.ones([3, 3])
        XCTAssertEqual(o.shape, [3, 3])
    }

    func testTensorArithmetic() {
        let a = MLXArray.ones([4])
        let b = a * 3 + 1
        eval(b)
        XCTAssertEqual(b[0].item(Float.self), 4.0, accuracy: 1e-6)
    }

    func testMatMul() {
        // [3, 4] @ [4, 5] → [3, 5]; each element = sum of 4 ones = 4.0
        let a = MLXArray.ones([3, 4])
        let b = MLXArray.ones([4, 5])
        let c = matmul(a, b)
        eval(c)
        XCTAssertEqual(c.shape, [3, 5])
        XCTAssertEqual(c[0, 0].item(Float.self), 4.0, accuracy: 1e-6)
    }

    func testSoftmax() {
        let x = MLXArray([1.0, 2.0, 3.0] as [Float])
        let sm = softmax(x, axis: -1)
        eval(sm)
        let total = sm.sum()
        eval(total)
        XCTAssertEqual(total.item(Float.self), 1.0, accuracy: 1e-5)
    }

    func testSqrtAndPow() {
        let x = MLXArray([4.0, 9.0, 16.0] as [Float])
        let s = sqrt(x)
        eval(s)
        XCTAssertEqual(s[0].item(Float.self), 2.0, accuracy: 1e-6)
        XCTAssertEqual(s[1].item(Float.self), 3.0, accuracy: 1e-6)
    }

    // MARK: - 2. Linear Layer
    // Required by: Q/K/V projections, FFN up/down, output projection

    func testLinearForwardShape() {
        let layer = Linear(4, 8)
        let x = MLXArray.zeros([2, 4])
        let y = layer(x)
        eval(y)
        XCTAssertEqual(y.shape, [2, 8])
    }

    func testLinearNoBias() {
        let layer = Linear(4, 8, bias: false)
        let x = MLXArray.zeros([2, 4])
        let y = layer(x)
        eval(y)
        XCTAssertEqual(y.shape, [2, 8])
    }

    func testLinearBatchedInput() {
        // [batch, seq, dim] → [batch, seq, out_dim]
        let layer = Linear(64, 128)
        let x = MLXArray.zeros([2, 10, 64])
        let y = layer(x)
        eval(y)
        XCTAssertEqual(y.shape, [2, 10, 128])
    }

    // MARK: - 3. Normalization Layers
    // Required by: pre/post norm in every DiT block and LM transformer block

    func testLayerNorm() {
        let norm = LayerNorm(dimensions: 64)
        let x = MLXRandom.normal([2, 10, 64])
        let y = norm(x)
        eval(y)
        XCTAssertEqual(y.shape, [2, 10, 64])
    }

    func testLayerNormOutputMeanNearZero() {
        let norm = LayerNorm(dimensions: 64, affine: false)
        let x = MLXRandom.normal([1, 1, 64])
        let y = norm(x)
        eval(y)
        let mean = y.mean()
        eval(mean)
        XCTAssertEqual(mean.item(Float.self), 0.0, accuracy: 1e-4)
    }

    func testRMSNorm() {
        let norm = RMSNorm(dimensions: 64)
        let x = MLXRandom.normal([2, 10, 64])
        let y = norm(x)
        eval(y)
        XCTAssertEqual(y.shape, [2, 10, 64])
    }

    // MARK: - 4. Embedding Layer
    // Required by: LM token embedding, lyric encoder token embedding

    func testEmbeddingLookup() {
        let embed = Embedding(embeddingCount: 1000, dimensions: 64)
        let indices = MLXArray([0, 1, 5, 42])
        let y = embed(indices)
        eval(y)
        XCTAssertEqual(y.shape, [4, 64])
    }

    func testEmbeddingBatched() {
        let embed = Embedding(embeddingCount: 500, dimensions: 32)
        // [batch=2, seq=8] token ids
        let indices = MLXArray.zeros([2, 8], dtype: .int32)
        let y = embed(indices)
        eval(y)
        XCTAssertEqual(y.shape, [2, 8, 32])
    }

    // MARK: - 5. Multi-Head Attention
    // Required by: DiT self-attention, DiT cross-attention (conditioning), LM self-attention
    // Note: MHA callAsFunction is `mha(_ queries:, keys:, values:)` — first arg has no label.

    func testMultiHeadAttentionSelfAttention() {
        let mha = MultiHeadAttention(dimensions: 64, numHeads: 8)
        let q = MLXArray.zeros([1, 10, 64])
        let k = MLXArray.zeros([1, 10, 64])
        let v = MLXArray.zeros([1, 10, 64])
        let out = mha(q, keys: k, values: v)
        eval(out)
        XCTAssertEqual(out.shape, [1, 10, 64])
    }

    func testMultiHeadAttentionCrossAttention() {
        // Cross-attention: queries from latents (seq=50), keys/values from conditioning (seq=77)
        let mha = MultiHeadAttention(
            dimensions: 64,
            numHeads: 8,
            queryInputDimensions: 64,
            keyInputDimensions: 128,
            valueInputDimensions: 128
        )
        let q = MLXArray.zeros([1, 50, 64])
        let k = MLXArray.zeros([1, 77, 128])
        let v = MLXArray.zeros([1, 77, 128])
        let out = mha(q, keys: k, values: v)
        eval(out)
        XCTAssertEqual(out.shape, [1, 50, 64])
    }

    // MARK: - 6. RoPE Positional Embeddings
    // Required by: DiT self-attention, LM self-attention
    // Note: RoPE callAsFunction is `rope(_ x:, offset:)` — offset is required (not defaulted).

    func testRoPEShape() {
        let rope = RoPE(dimensions: 64)
        // RoPE operates on the last two dims: [*, seq, dim]
        let x = MLXArray.zeros([1, 10, 64])
        let y = rope(x, offset: 0)
        eval(y)
        XCTAssertEqual(y.shape, [1, 10, 64])
    }

    func testRoPEWithOffset() {
        let rope = RoPE(dimensions: 64)
        let x = MLXArray.zeros([1, 5, 64])
        let y = rope(x, offset: 10) // absolute positions 10..14
        eval(y)
        XCTAssertEqual(y.shape, [1, 5, 64])
    }

    // MARK: - 7. Conv1d (AudioVAE)
    // Required by: VAE encoder (audio → latent) and decoder (latent → audio)
    // mlx-swift Conv1d uses NLC layout: [batch, length, channels]

    func testConv1dShape() {
        // Same-padding (padding = kernelSize/2): output length matches input length
        let conv = Conv1d(inputChannels: 8, outputChannels: 16, kernelSize: 3, padding: 1)
        let x = MLXArray.zeros([1, 100, 8]) // [batch, length, channels]
        let y = conv(x)
        eval(y)
        XCTAssertEqual(y.shape[0], 1)
        XCTAssertEqual(y.shape[1], 100) // length preserved with padding=1, kernel=3
        XCTAssertEqual(y.shape[2], 16)
    }

    func testConv1dStrided() {
        // Stride 2 halves the sequence length (downsampling, as in VAE encoder)
        let conv = Conv1d(inputChannels: 4, outputChannels: 8, kernelSize: 4, stride: 2, padding: 1)
        let x = MLXArray.zeros([1, 64, 4])
        let y = conv(x)
        eval(y)
        XCTAssertEqual(y.shape[2], 8)
        XCTAssertLessThan(y.shape[1], 64) // length decreases with stride > 1
    }

    // MARK: - 8. DDIM / Flow-Matching Sampler Step (raw MLX ops)
    // mlx-swift has no built-in diffusion sampler; TurboSampler will be implemented
    // from primitives. These tests validate that the required ops produce numerically
    // stable results at the scales used by ACE-Step (latent dim ~64, seq ~256).

    func testDDIMStepNumericalStability() {
        let batchSize = 1
        let seqLen = 256
        let latentDim = 64

        // DDIM: x_{t-1} = sqrt(α_{t-1}) * x0_pred + sqrt(1 - α_{t-1}) * ε
        let alphaT: Float = 0.9
        let alphaPrev: Float = 0.95

        let xt = MLXRandom.normal([batchSize, seqLen, latentDim])
        let eps = MLXRandom.normal([batchSize, seqLen, latentDim])

        let sqrtAlphaT    = MLXArray(Float(alphaT).squareRoot())
        let sqrtOneMinusT = MLXArray(Float(1.0 - alphaT).squareRoot())
        let sqrtAlphaPrev = MLXArray(Float(alphaPrev).squareRoot())
        let sqrtOneMinusP = MLXArray(Float(1.0 - alphaPrev).squareRoot())

        let x0Pred = (xt - sqrtOneMinusT * eps) / sqrtAlphaT
        let xPrev  = sqrtAlphaPrev * x0Pred + sqrtOneMinusP * eps

        eval(xPrev)
        XCTAssertEqual(xPrev.shape, [batchSize, seqLen, latentDim])

        let s = xPrev.sum()
        eval(s)
        XCTAssertTrue(s.item(Float.self).isFinite, "DDIM step produced non-finite values")
    }

    func testFlowMatchingEulerStep() {
        // Euler step for flow-matching: x_{t+dt} = x_t + dt * v(x_t, t)
        let x = MLXRandom.normal([1, 256, 64])
        let v = MLXRandom.normal([1, 256, 64])
        let dt = MLXArray(Float(1.0 / 8.0))

        let xNext = x + dt * v
        eval(xNext)

        XCTAssertEqual(xNext.shape, [1, 256, 64])
        let s = xNext.sum()
        eval(s)
        XCTAssertTrue(s.item(Float.self).isFinite)
    }

    // MARK: - 9. Safetensors Weight Loading
    // Required by: DiTWeightLoader, LMWeightLoader
    // Writes a synthetic .safetensors file and verifies mlx-swift loads it correctly.
    // Confirms: loadArrays(url:) works, shapes match, values are bit-exact.

    func testSafetensorsRoundTrip() throws {
        // Build a minimal valid .safetensors file with one float32 tensor.
        // Format spec: 8-byte LE uint64 header_size || JSON header || raw tensor data
        let tensorName = "model.layers.0.weight"
        let shape = [2, 3]
        let values: [Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        let byteCount = values.count * MemoryLayout<Float32>.size // 24

        let headerJSON = """
        {"\(tensorName)":{"dtype":"F32","shape":\(shape),"data_offsets":[0,\(byteCount)]}}
        """
        var headerBytes = Array(headerJSON.utf8)
        // Pad to 8-byte boundary with spaces (valid per spec)
        while headerBytes.count % 8 != 0 { headerBytes.append(0x20) }

        var file = Data()
        var headerSize = UInt64(headerBytes.count).littleEndian
        file.append(Data(bytes: &headerSize, count: 8))
        file.append(Data(headerBytes))
        file.append(values.withUnsafeBytes { Data($0) })

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("probe_\(UUID().uuidString).safetensors")
        try file.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let weights = try loadArrays(url: url)
        guard let tensor = weights[tensorName] else {
            XCTFail("Expected key '\(tensorName)' not found in loaded safetensors")
            return
        }
        eval(tensor)

        XCTAssertEqual(tensor.shape, [2, 3])
        XCTAssertEqual(tensor.dtype, .float32)
        XCTAssertEqual(tensor[0, 0].item(Float.self), 1.0, accuracy: 1e-6)
        XCTAssertEqual(tensor[1, 2].item(Float.self), 6.0, accuracy: 1e-6)
    }

    // MARK: - 10. SwiGLU / GELU Activation (FFN gating)
    // Required by: DiT FFN blocks, LM FFN blocks

    func testSwiGLUGating() {
        // SwiGLU: FFN(x) = SiLU(gate) * value, both halves of a 2x projection
        let proj = Linear(64, 128)
        let x = MLXRandom.normal([1, 10, 64])
        let projected = proj(x)
        eval(projected)
        XCTAssertEqual(projected.shape, [1, 10, 128])

        let gate  = projected[.ellipsis, ..<64]
        let value = projected[.ellipsis, 64...]

        let activated = silu(gate) * value
        eval(activated)
        XCTAssertEqual(activated.shape, [1, 10, 64])
    }

    func testGELUActivation() {
        let x = MLXArray([-2.0, -1.0, 0.0, 1.0, 2.0] as [Float])
        let y = gelu(x)
        eval(y)
        XCTAssertEqual(y.shape, [5])
        XCTAssertEqual(y[2].item(Float.self), 0.0, accuracy: 1e-4) // gelu(0) = 0
        XCTAssertLessThan(y[1].item(Float.self), y[3].item(Float.self)) // monotone
    }
}

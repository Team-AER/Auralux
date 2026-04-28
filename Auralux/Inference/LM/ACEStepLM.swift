import MLX
import MLXNN
import MLXRandom
import Foundation

// MARK: - Config

struct ACEStepLMConfig: Codable, Sendable {
    var vocabSize: Int
    var hiddenSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var intermediateSize: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var maxPositionEmbeddings: Int

    var headDim: Int = 128

    init(
        vocabSize: Int = 217204,
        hiddenSize: Int = 1024,
        numHiddenLayers: Int = 28,
        numAttentionHeads: Int = 16,
        numKeyValueHeads: Int = 8,
        headDim: Int = 128,
        intermediateSize: Int = 3072,
        rmsNormEps: Float = 1e-6,
        ropeTheta: Float = 1_000_000.0,
        maxPositionEmbeddings: Int = 32768
    ) {
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.intermediateSize = intermediateSize
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.maxPositionEmbeddings = maxPositionEmbeddings
    }

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case intermediateSize = "intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
    }
}

// MARK: - Causal mask

/// Returns an additive [seqLen, seqLen] mask: 0 for past/diagonal, -1e9 for future.
/// Works with integer (0,1,2,...) row/col positions so binary = future - relu(future-1) is exact.
private func makeCausalMask(seqLen: Int) -> MLXArray? {
    guard seqLen > 1 else { return nil }
    let i = MLXArray(Array(0..<seqLen).map { Float($0) }).reshaped([seqLen, 1])
    let j = MLXArray(Array(0..<seqLen).map { Float($0) }).reshaped([1, seqLen])
    let future = relu(j - i)                               // 0 for j<=i, j-i for j>i
    let binary  = future - relu(future - MLXArray(Float(1.0))) // clamp to {0, 1}
    return binary * Float(-1e9)
}

// MARK: - GQA (Grouped Query Attention) helper

/// Tile k/v from [B, numKVHeads, L, D] to [B, numHeads, L, D] with correct grouped ordering.
/// Each KV head kvi serves Q heads [kvi*g .. kvi*g+g-1].
private func gqaTile(_ x: MLXArray, fromKV numKVHeads: Int, toQ numHeads: Int) -> MLXArray {
    guard numKVHeads < numHeads else { return x }
    let g = numHeads / numKVHeads
    let B = x.shape[0], L = x.shape[2], D = x.shape[3]
    // Concatenate g copies: [B, g*numKVHeads, L, D] in round-robin order
    let tiled = concatenated(Array(repeating: x, count: g), axis: 1)
    // Reshape→transpose→reshape to achieve grouped ordering [k0,k0,..,k1,k1,..]
    return tiled
        .reshaped([B, g, numKVHeads, L, D])
        .transposed(0, 2, 1, 3, 4)
        .reshaped([B, numHeads, L, D])
}

// MARK: - Attention (GQA + RoPE + Q/K norm)

final class ACEStepLMAttention: Module {
    let qProj: Linear
    let kProj: Linear
    let vProj: Linear
    let oProj: Linear
    let qNorm: RMSNorm
    let kNorm: RMSNorm
    let rope: RoPE
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    init(config: ACEStepLMConfig) {
        let hd = config.headDim
        numHeads    = config.numAttentionHeads
        numKVHeads  = config.numKeyValueHeads
        headDim     = hd
        scale       = 1.0 / Float(hd).squareRoot()
        qProj = Linear(config.hiddenSize, config.numAttentionHeads * hd, bias: false)
        kProj = Linear(config.hiddenSize, config.numKeyValueHeads * hd, bias: false)
        vProj = Linear(config.hiddenSize, config.numKeyValueHeads * hd, bias: false)
        oProj = Linear(config.numAttentionHeads * hd, config.hiddenSize, bias: false)
        qNorm = RMSNorm(dimensions: hd, eps: config.rmsNormEps)
        kNorm = RMSNorm(dimensions: hd, eps: config.rmsNormEps)
        rope  = RoPE(dimensions: hd, base: config.ropeTheta)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, offset: Int = 0, mask: MLXArray? = nil) -> MLXArray {
        let B = x.shape[0], L = x.shape[1]

        // Project → [B, L, H, D]
        var q = qProj(x).reshaped([B, L, numHeads, headDim])
        var k = kProj(x).reshaped([B, L, numKVHeads, headDim])
        var v = vProj(x).reshaped([B, L, numKVHeads, headDim])

        // Per-head RMS norm (over D)
        q = qNorm(q)
        k = kNorm(k)

        // Apply RoPE: rope operates on [*, seq, dim]; merge B+H for the call
        q = q.transposed(0, 2, 1, 3).reshaped([B * numHeads, L, headDim])
        q = rope(q, offset: offset).reshaped([B, numHeads, L, headDim])

        k = k.transposed(0, 2, 1, 3).reshaped([B * numKVHeads, L, headDim])
        k = rope(k, offset: offset).reshaped([B, numKVHeads, L, headDim])
        v = v.transposed(0, 2, 1, 3)  // [B, numKVHeads, L, D]

        // GQA tile KV heads to match Q
        k = gqaTile(k, fromKV: numKVHeads, toQ: numHeads)
        v = gqaTile(v, fromKV: numKVHeads, toQ: numHeads)

        // Scaled dot-product: [B, H, L, L]. fp32 softmax for numerical stability
        // (fp16 overflows on long sequences).
        let origDtype = q.dtype
        var scores = matmul(q, k.transposed(0, 1, 3, 2)) * scale
        if let m = mask {
            scores = scores + m
        }
        scores = softmax(scores.asType(.float32), axis: -1).asType(origDtype)

        // Weighted sum → [B, L, H*D]
        let out = matmul(scores, v)
            .transposed(0, 2, 1, 3)
            .reshaped([B, L, numHeads * headDim])

        return oProj(out)
    }
}

// MARK: - SwiGLU FFN

final class ACEStepLMFFN: Module {
    let gateProj: Linear
    let upProj: Linear
    let downProj: Linear

    init(config: ACEStepLMConfig) {
        gateProj = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        upProj   = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        downProj = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Transformer Block

final class ACEStepLMBlock: Module {
    let inputLayernorm: RMSNorm
    let selfAttn: ACEStepLMAttention
    let postAttentionLayernorm: RMSNorm
    let mlp: ACEStepLMFFN

    init(config: ACEStepLMConfig) {
        inputLayernorm         = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        selfAttn               = ACEStepLMAttention(config: config)
        postAttentionLayernorm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        mlp                    = ACEStepLMFFN(config: config)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, offset: Int = 0, mask: MLXArray? = nil) -> MLXArray {
        var h = x + selfAttn(inputLayernorm(x), offset: offset, mask: mask)
        h = h + mlp(postAttentionLayernorm(h))
        return h
    }
}

// MARK: - Full Model

final class ACEStepLMModel: Module, @unchecked Sendable {
    let embedTokens: Embedding
    let layers: [ACEStepLMBlock]
    let norm: RMSNorm
    let lmHead: Linear
    let config: ACEStepLMConfig

    init(config: ACEStepLMConfig = ACEStepLMConfig()) {
        self.config  = config
        embedTokens  = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        layers       = (0..<config.numHiddenLayers).map { _ in ACEStepLMBlock(config: config) }
        norm         = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        lmHead       = Linear(config.hiddenSize, config.vocabSize, bias: false)
        super.init()
    }

    /// Forward: returns logits [B, L, vocabSize].
    func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        let L = inputIds.shape[1]
        var x = embedTokens(inputIds)
        let mask = makeCausalMask(seqLen: L)
        for layer in layers {
            x = layer(x, mask: mask)
        }
        return lmHead(norm(x))
    }

    /// Returns last-layer hidden states [B, L, hiddenSize] without projecting to vocab.
    /// Used as input to AceLyricEncoder for text conditioning.
    func hiddenStates(_ inputIds: MLXArray) -> MLXArray {
        let L = inputIds.shape[1]
        var x = embedTokens(inputIds)
        let mask = makeCausalMask(seqLen: L)
        for layer in layers {
            x = layer(x, mask: mask)
        }
        return norm(x)
    }

    /// Greedy autoregressive generation (synchronous; no KV-cache).
    /// Returns the generated token ids (excluding the prompt).
    func generate(inputIds: [Int], maxNewTokens: Int, temperature: Float = 1.0) -> [Int] {
        var allIds = inputIds
        var generated: [Int] = []

        for _ in 0..<maxNewTokens {
            let input = MLXArray(allIds).reshaped([1, allIds.count])
            let logits = callAsFunction(input)          // [1, L, vocabSize]
            eval(logits)

            let lastLogits = logits[0, allIds.count - 1, 0...]  // [vocabSize]
            let nextToken: Int
            if temperature > 0 {
                let scaled = lastLogits / MLXArray(temperature)
                let probs  = softmax(scaled, axis: -1)
                eval(probs)
                nextToken = argMax(probs, axis: -1).item(Int.self)
            } else {
                nextToken = argMax(lastLogits, axis: -1).item(Int.self)
            }

            generated.append(nextToken)
            allIds.append(nextToken)
        }

        return generated
    }
}

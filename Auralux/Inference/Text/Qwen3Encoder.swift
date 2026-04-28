import MLX
import MLXNN
import Foundation

// Qwen3-Embedding-0.6B — bidirectional encoder used by ACE-Step v1.5 Turbo
// for text/lyric conditioning. Architecture identical to Qwen3 base
// (RMSNorm + GQA + RoPE + SwiGLU MLP), but no causal mask and no lm_head.
//
// Two distinct call modes per upstream `conditioning_embed.py`:
//   • `embed(input_ids)`  → embedding-table lookup only (used for lyrics)
//   • `encode(input_ids)` → full 28-layer bidirectional forward pass (used for text)
//
// Checkpoint keys (after `tools/convert_weights.py`):
//   embedTokens.weight                              [151669, 1024]
//   layers.{i}.inputLayernorm.weight                [1024]
//   layers.{i}.selfAttn.{q,k,v,o}Proj.weight        [..]
//   layers.{i}.selfAttn.{q,k}Norm.weight            [128]   (per-head)
//   layers.{i}.postAttentionLayernorm.weight        [1024]
//   layers.{i}.mlp.{gate,up,down}Proj.weight        [..]
//   norm.weight                                     [1024]

// MARK: - Config

struct Qwen3EncoderConfig: Sendable {
    let vocabSize:        Int
    let hiddenSize:       Int
    let numHiddenLayers:  Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim:          Int
    let intermediateSize: Int
    let rmsNormEps:       Float
    let ropeTheta:        Float

    init(
        vocabSize:         Int   = 151669,
        hiddenSize:        Int   = 1024,
        numHiddenLayers:   Int   = 28,
        numAttentionHeads: Int   = 16,
        numKeyValueHeads:  Int   = 8,
        headDim:           Int   = 128,
        intermediateSize:  Int   = 3072,
        rmsNormEps:        Float = 1e-6,
        ropeTheta:         Float = 1_000_000.0
    ) {
        self.vocabSize         = vocabSize
        self.hiddenSize        = hiddenSize
        self.numHiddenLayers   = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads  = numKeyValueHeads
        self.headDim           = headDim
        self.intermediateSize  = intermediateSize
        self.rmsNormEps        = rmsNormEps
        self.ropeTheta         = ropeTheta
    }
}

// MARK: - GQA helper (same semantics as ACEStepLM.gqaTile)

private func gqaTile(_ x: MLXArray, fromKV numKVHeads: Int, toQ numHeads: Int) -> MLXArray {
    guard numKVHeads < numHeads else { return x }
    let g = numHeads / numKVHeads
    return repeated(x, count: g, axis: 1)
}

// MARK: - Bidirectional attention with GQA + per-head Q/K norm + RoPE

final class Qwen3EncoderAttention: Module, @unchecked Sendable {
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

    init(config: Qwen3EncoderConfig) {
        let hd = config.headDim
        numHeads   = config.numAttentionHeads
        numKVHeads = config.numKeyValueHeads
        headDim    = hd
        scale      = 1.0 / Float(hd).squareRoot()
        qProj = Linear(config.hiddenSize, config.numAttentionHeads * hd, bias: false)
        kProj = Linear(config.hiddenSize, config.numKeyValueHeads  * hd, bias: false)
        vProj = Linear(config.hiddenSize, config.numKeyValueHeads  * hd, bias: false)
        oProj = Linear(config.numAttentionHeads * hd, config.hiddenSize, bias: false)
        qNorm = RMSNorm(dimensions: hd, eps: config.rmsNormEps)
        kNorm = RMSNorm(dimensions: hd, eps: config.rmsNormEps)
        rope  = RoPE(dimensions: hd, base: config.ropeTheta)
        super.init()
    }

    /// Bidirectional self-attention. `mask` is an optional additive `[*, S, S]` tensor
    /// (e.g. padding mask). Caller passes `nil` for full bidirectional attention.
    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let B = x.shape[0], L = x.shape[1]

        var q = qProj(x).reshaped([B, L, numHeads,   headDim])
        var k = kProj(x).reshaped([B, L, numKVHeads, headDim])
        var v = vProj(x).reshaped([B, L, numKVHeads, headDim])

        q = qNorm(q)
        k = kNorm(k)

        // [B, H, L, D]
        q = q.transposed(0, 2, 1, 3).reshaped([B * numHeads,   L, headDim])
        q = rope(q).reshaped([B, numHeads,   L, headDim])
        k = k.transposed(0, 2, 1, 3).reshaped([B * numKVHeads, L, headDim])
        k = rope(k).reshaped([B, numKVHeads, L, headDim])
        v = v.transposed(0, 2, 1, 3)

        k = gqaTile(k, fromKV: numKVHeads, toQ: numHeads)
        v = gqaTile(v, fromKV: numKVHeads, toQ: numHeads)

        let origDtype = q.dtype
        var scores = matmul(q, k.transposed(0, 1, 3, 2)) * scale
        if let m = mask {
            scores = scores + m
        }
        // fp32 softmax for numerical stability — fp16 overflows for long sequences.
        scores = softmax(scores.asType(.float32), axis: -1).asType(origDtype)

        let out = matmul(scores, v)
            .transposed(0, 2, 1, 3)
            .reshaped([B, L, numHeads * headDim])
        return oProj(out)
    }
}

// MARK: - SwiGLU MLP (same as Qwen3 base)

final class Qwen3EncoderFFN: Module, @unchecked Sendable {
    let gateProj: Linear
    let upProj:   Linear
    let downProj: Linear

    init(config: Qwen3EncoderConfig) {
        gateProj = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        upProj   = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        downProj = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Block (RMSNorm-pre + bidirectional self-attn + RMSNorm-pre + MLP)

final class Qwen3EncoderBlock: Module, @unchecked Sendable {
    let inputLayernorm:         RMSNorm
    let selfAttn:               Qwen3EncoderAttention
    let postAttentionLayernorm: RMSNorm
    let mlp:                    Qwen3EncoderFFN

    init(config: Qwen3EncoderConfig) {
        inputLayernorm         = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        selfAttn               = Qwen3EncoderAttention(config: config)
        postAttentionLayernorm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        mlp                    = Qwen3EncoderFFN(config: config)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var h = x + selfAttn(inputLayernorm(x), mask: mask)
        h = h + mlp(postAttentionLayernorm(h))
        return h
    }
}

// MARK: - Top-level encoder

final class Qwen3EncoderModel: Module, @unchecked Sendable {
    let embedTokens: Embedding
    let layers: [Qwen3EncoderBlock]
    let norm:   RMSNorm
    let config: Qwen3EncoderConfig

    init(config: Qwen3EncoderConfig = Qwen3EncoderConfig()) {
        self.config = config
        embedTokens = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        layers      = (0..<config.numHiddenLayers).map { _ in Qwen3EncoderBlock(config: config) }
        norm        = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    /// Embedding-table lookup only — the path used for lyric conditioning per
    /// `conditioning_embed.py:76-79`.
    /// - inputIds: `[B, S]` token IDs.
    /// - returns: `[B, S, hiddenSize]` raw embeddings (no transformer layers run).
    func embed(_ inputIds: MLXArray) -> MLXArray {
        embedTokens(inputIds)
    }

    /// Full **causal** encoder forward pass — the path used for text conditioning per
    /// `conditioning_embed.py:71-74`. Qwen3-Embedding-0.6B is `Qwen3ForCausalLM` and
    /// `AutoModel.from_pretrained` returns the causal `Qwen3Model`; ablating the
    /// causal mask diverges from upstream by ~3× in peak hidden-state magnitude.
    ///
    /// - inputIds: `[B, S]` token IDs.
    /// - extraMask: optional additional additive mask (e.g. for padding); usually `nil`.
    /// - returns: `[B, S, hiddenSize]` last hidden states (post final RMSNorm).
    ///
    /// Runs in fp32 internally — q_norm/k_norm have large weights (max ~96) and 28
    /// layers of accumulated fp16 precision loss diverge from upstream.
    func encode(_ inputIds: MLXArray, extraMask: MLXArray? = nil) -> MLXArray {
        let S = inputIds.shape.last ?? 0
        var x = embedTokens(inputIds).asType(.float32)
        let causal = Self.causalMask(seqLen: S, dtype: x.dtype)
        let mask: MLXArray
        if let extraMask {
            mask = causal != nil ? (causal! + extraMask) : extraMask
        } else {
            mask = causal ?? MLXArray.zeros([1, 1, max(S, 1), max(S, 1)])
        }
        for layer in layers {
            x = layer(x, mask: mask)
        }
        return norm(x)
    }

    /// Build an additive `[1, 1, S, S]` mask: 0 for j<=i (allowed), -1e9 for j>i (masked).
    /// Returns `nil` if `S <= 1` (no masking needed).
    private static func causalMask(seqLen S: Int, dtype: DType) -> MLXArray? {
        guard S > 1 else { return nil }
        let i = MLXArray(Array(0..<S).map { Float($0) }).reshaped([S, 1])
        let j = MLXArray(Array(0..<S).map { Float($0) }).reshaped([1, S])
        let future = relu(j - i)                                  // 0 if j<=i, j-i if j>i
        let binary = future - relu(future - MLXArray(Float(1.0))) // clamp to {0, 1}
        let m = (binary * Float(-1e9)).asType(dtype)
        return m.reshaped([1, 1, S, S])
    }
}

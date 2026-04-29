import MLX
import MLXNN
import MLXRandom
import Foundation

// MARK: - Config (ACE-Step v1.5 Turbo — configuration_acestep_v15.py)

struct AceStepConfig: Sendable {
    let hiddenSize: Int
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let intermediateSize: Int
    let numDiTLayers: Int
    let numLyricEncoderLayers: Int
    let numTimbreEncoderLayers: Int
    let numDetokenizerLayers: Int
    let numAttentionPoolerLayers: Int
    let patchSize: Int
    let poolWindowSize: Int
    let audioAcousticHiddenDim: Int
    let timbreHiddenDim: Int
    let inChannels: Int
    let rmsNormEps: Float
    let ropeTheta: Float
    let textHiddenDim: Int
    let freqDim: Int
    let slidingWindow: Int
    let useSlidingWindow: Bool

    // FSQ / audio tokenizer configuration
    let fsqDim: Int
    let fsqInputLevels: [Int]
    let fsqInputNumQuantizers: Int

    // Encoder-specific dimensions (default to main DiT dimensions when nil).
    // XL variants have a smaller encoder stack than the DiT decoder.
    let encoderHiddenSize: Int
    let encoderNumHeads: Int
    let encoderNumKVHeads: Int
    let encoderIntermediateSize: Int

    /// True for the turbo checkpoint (CFG distilled into weights via `time_embed_r`).
    /// False for SFT / base (standard two-pass CFG at inference time).
    let usesCFGDistillation: Bool

    init(
        usesCFGDistillation: Bool    = true,
        hiddenSize: Int              = 2048,
        numHeads: Int                = 16,
        numKVHeads: Int              = 8,
        headDim: Int                 = 128,
        intermediateSize: Int        = 6144,
        numDiTLayers: Int            = 24,
        numLyricEncoderLayers: Int   = 8,
        numTimbreEncoderLayers: Int  = 4,
        numDetokenizerLayers: Int    = 2,
        numAttentionPoolerLayers: Int = 2,
        patchSize: Int               = 2,
        poolWindowSize: Int          = 5,
        audioAcousticHiddenDim: Int  = 64,
        timbreHiddenDim: Int         = 64,
        inChannels: Int              = 192,
        rmsNormEps: Float            = 1e-6,
        ropeTheta: Float             = 1_000_000.0,
        textHiddenDim: Int           = 1024,
        freqDim: Int                 = 256,
        slidingWindow: Int           = 128,
        useSlidingWindow: Bool       = true,
        fsqDim: Int                  = 6,
        fsqInputLevels: [Int]        = [8, 8, 8, 5, 5, 5],
        fsqInputNumQuantizers: Int   = 1,
        encoderHiddenSize: Int?      = nil,
        encoderNumHeads: Int?        = nil,
        encoderNumKVHeads: Int?      = nil,
        encoderIntermediateSize: Int? = nil
    ) {
        self.usesCFGDistillation    = usesCFGDistillation
        self.hiddenSize             = hiddenSize
        self.numHeads               = numHeads
        self.numKVHeads             = numKVHeads
        self.headDim                = headDim
        self.intermediateSize       = intermediateSize
        self.numDiTLayers           = numDiTLayers
        self.numLyricEncoderLayers  = numLyricEncoderLayers
        self.numTimbreEncoderLayers = numTimbreEncoderLayers
        self.numDetokenizerLayers   = numDetokenizerLayers
        self.numAttentionPoolerLayers = numAttentionPoolerLayers
        self.patchSize              = patchSize
        self.poolWindowSize         = poolWindowSize
        self.audioAcousticHiddenDim = audioAcousticHiddenDim
        self.timbreHiddenDim        = timbreHiddenDim
        self.inChannels             = inChannels
        self.rmsNormEps             = rmsNormEps
        self.ropeTheta              = ropeTheta
        self.textHiddenDim          = textHiddenDim
        self.freqDim                = freqDim
        self.slidingWindow          = slidingWindow
        self.useSlidingWindow       = useSlidingWindow
        self.fsqDim                 = fsqDim
        self.fsqInputLevels         = fsqInputLevels
        self.fsqInputNumQuantizers   = fsqInputNumQuantizers
        self.encoderHiddenSize       = encoderHiddenSize       ?? hiddenSize
        self.encoderNumHeads         = encoderNumHeads         ?? numHeads
        self.encoderNumKVHeads       = encoderNumKVHeads       ?? numKVHeads
        self.encoderIntermediateSize = encoderIntermediateSize ?? intermediateSize
    }

    static let turbo   = AceStepConfig(usesCFGDistillation: true)
    static let sft     = AceStepConfig(usesCFGDistillation: false)
    static let base    = AceStepConfig(usesCFGDistillation: false)
    static let xlTurbo = AceStepConfig(
        usesCFGDistillation: true,
        hiddenSize: 2560, numHeads: 32, numKVHeads: 8,
        intermediateSize: 9728, numDiTLayers: 32,
        encoderHiddenSize: 2048, encoderNumHeads: 16,
        encoderNumKVHeads: 8, encoderIntermediateSize: 6144
    )
    static let xlSft   = AceStepConfig(
        usesCFGDistillation: false,
        hiddenSize: 2560, numHeads: 32, numKVHeads: 8,
        intermediateSize: 9728, numDiTLayers: 32,
        encoderHiddenSize: 2048, encoderNumHeads: 16,
        encoderNumKVHeads: 8, encoderIntermediateSize: 6144
    )
    static let xlBase  = AceStepConfig(
        usesCFGDistillation: false,
        hiddenSize: 2560, numHeads: 32, numKVHeads: 8,
        intermediateSize: 9728, numDiTLayers: 32,
        encoderHiddenSize: 2048, encoderNumHeads: 16,
        encoderNumKVHeads: 8, encoderIntermediateSize: 6144
    )

    // Per-layer attention type — mirrors `configuration_acestep_v15.py:251-254`:
    //   layer_types[i] = "sliding_attention" if (i+1)%2 else "full_attention"
    // i.e. even-index layers are sliding, odd-index layers are full.
    func attentionType(for layerIndex: Int) -> String {
        layerIndex % 2 == 0 ? "sliding_attention" : "full_attention"
    }
}

// MARK: - Sinusoidal timestep embedding

private func sinusoidalEmbedding(t: MLXArray, dim: Int) -> MLXArray {
    let half  = dim / 2
    let freqs = exp(
        MLXArray(Array(0..<half).map { Float($0) }) * (-log(10_000.0) / Float(half))
    )
    let tCol = t.reshaped([-1, 1]).asType(.float32)
    let args = tCol * freqs.reshaped([1, -1])
    return concatenated([cos(args), sin(args)], axis: -1)
}

// MARK: - Attention mask helpers
//
// Match upstream `create_4d_mask` (modeling_acestep_v15_turbo.py:53-132):
// bidirectional, optionally sliding-window. Returns additive masks (0 = keep,
// -1e9 = block) broadcast-compatible with `[B, H, L_q, L_kv]` attention scores.

/// Sliding-window mask `[1, 1, L, L]` for bidirectional attention with window `W`:
/// `0` where `|i - j| <= W`, `-1e9` otherwise. Returns `nil` for trivial sequences.
private func slidingWindowMask(seqLen L: Int, window W: Int) -> MLXArray? {
    guard L > 1, W >= 0 else { return nil }
    let i = MLXArray(Array(0..<L).map { Float($0) }).reshaped([L, 1])
    let j = MLXArray(Array(0..<L).map { Float($0) }).reshaped([1, L])
    // exceeded > 0 when |i-j| > W; clip to {0,1} via the same trick used in causalMask.
    let exceeded = relu(abs(i - j) - Float(W))
    let binary   = exceeded - relu(exceeded - Float(1.0))   // {0, 1}
    return (binary * Float(-1e9)).reshaped([1, 1, L, L])
}

/// Convert a `[B, S]` int padding mask (1 = valid, 0 = pad) into an additive `[B, 1, 1, S]`
/// key-padding mask. Broadcasts against `[B, H, L_q, S]` cross-attn scores and `[B, H, S, S]`
/// self-attn scores.
private func keyPaddingMask(_ pad: MLXArray) -> MLXArray {
    let invalid = Float(1.0) - pad.asType(.float32)
    return (invalid * Float(-1e9)).reshaped([pad.shape[0], 1, 1, pad.shape[1]])
}

// MARK: - Timestep Embedder
// linear1 + linear2 produce the per-sample embedding (temb).
// timeProj produces the per-layer modulation vector (6×hiddenSize split into 6 chunks).

final class TimestepEmbedder: Module, @unchecked Sendable {
    let linear1: Linear
    let linear2: Linear
    let timeProj: Linear

    init(freqDim: Int = 256, hiddenSize: Int = 2048) {
        linear1  = Linear(freqDim,    hiddenSize,     bias: true)
        linear2  = Linear(hiddenSize, hiddenSize,     bias: true)
        timeProj = Linear(hiddenSize, hiddenSize * 6, bias: true)
        super.init()
    }

    // Returns (temb [B, H], timestepProj [B, 6, H])
    func callAsFunction(_ t: MLXArray) -> (MLXArray, MLXArray) {
        let freqDim = linear1.weight.shape[1]
        let tScaled = t * 1000.0
        let tFreq   = sinusoidalEmbedding(t: tScaled, dim: freqDim)  // t ∈ [0,1], scaled to 1000
        let temb    = linear2(silu(linear1(tFreq.asType(t.dtype))))
        let proj    = timeProj(silu(temb))
        let H       = temb.shape[1]
        return (temb, proj.reshaped([-1, 6, H]))
    }
}

// MARK: - Attention (GQA + per-head QK norms + optional RoPE)

final class AceStepAttention: Module, @unchecked Sendable {
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
    let isCrossAttention: Bool

    init(
        hiddenSize: Int, numHeads: Int, numKVHeads: Int, headDim: Int,
        rmsNormEps: Float, ropeTheta: Float, isCrossAttention: Bool = false
    ) {
        self.numHeads        = numHeads
        self.numKVHeads      = numKVHeads
        self.headDim         = headDim
        self.scale           = 1.0 / Float(headDim).squareRoot()
        self.isCrossAttention = isCrossAttention

        qProj = Linear(hiddenSize, numHeads   * headDim, bias: false)
        kProj = Linear(hiddenSize, numKVHeads * headDim, bias: false)
        vProj = Linear(hiddenSize, numKVHeads * headDim, bias: false)
        oProj = Linear(numHeads   * headDim, hiddenSize, bias: false)
        qNorm = RMSNorm(dimensions: headDim, eps: rmsNormEps)
        kNorm = RMSNorm(dimensions: headDim, eps: rmsNormEps)
        rope  = RoPE(dimensions: headDim, base: ropeTheta)
        super.init()
    }

    /// `mask` is an additive 4D tensor (broadcast-compatible with `[B, H, L, S]`).
    /// `nil` → no masking (full bidirectional).
    func callAsFunction(
        _ hidden: MLXArray,
        encoderHidden: MLXArray? = nil,
        mask: MLXArray? = nil,
        offset: Int = 0
    ) -> MLXArray {
        let B  = hidden.shape[0]
        let L  = hidden.shape[1]
        let kv = encoderHidden ?? hidden
        let S  = kv.shape[1]

        var q = qProj(hidden).reshaped([B, L, numHeads,   headDim])
        var k = kProj(kv    ).reshaped([B, S, numKVHeads, headDim])
        var v = vProj(kv    ).reshaped([B, S, numKVHeads, headDim])

        q = qNorm(q)
        k = kNorm(k)

        // [B, H, L, D]
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        if !isCrossAttention {
            let q2 = q.reshaped([B * numHeads,   L, headDim])
            let k2 = k.reshaped([B * numKVHeads, S, headDim])
            q = rope(q2, offset: offset).reshaped([B, numHeads,   L, headDim])
            k = rope(k2, offset: offset).reshaped([B, numKVHeads, S, headDim])
        }

        // GQA: tile KV heads to match Q heads
        if numKVHeads < numHeads {
            let g = numHeads / numKVHeads
            k = repeated(k, count: g, axis: 1)
            v = repeated(v, count: g, axis: 1)
        }

        // fp32 softmax for numerical stability — fp16 softmax can produce NaN on
        // long sequences when pre-softmax scores accumulate near fp16 max.
        let origDtype = q.dtype
        var w = matmul(q, k.transposed(0, 1, 3, 2)) * scale
        if let m = mask {
            w = w + m.asType(w.dtype)
        }
        w = softmax(w.asType(.float32), axis: -1).asType(origDtype)

        let out = matmul(w, v)
            .transposed(0, 2, 1, 3)
            .reshaped([B, L, numHeads * headDim])
        return oProj(out)
    }
}

// MARK: - SwiGLU MLP

final class SwiGLUMLP: Module, @unchecked Sendable {
    let gateProj: Linear
    let upProj: Linear
    let downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        gateProj = Linear(hiddenSize,       intermediateSize, bias: false)
        upProj   = Linear(hiddenSize,       intermediateSize, bias: false)
        downProj = Linear(intermediateSize, hiddenSize,       bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Encoder Layer (used by lyric encoder and detokenizer)

final class AceStepEncoderLayer: Module, @unchecked Sendable {
    let inputLayernorm: RMSNorm
    let selfAttn: AceStepAttention
    let postAttentionLayernorm: RMSNorm
    let mlp: SwiGLUMLP

    init(config: AceStepConfig) {
        inputLayernorm         = RMSNorm(dimensions: config.encoderHiddenSize, eps: config.rmsNormEps)
        selfAttn               = AceStepAttention(
            hiddenSize: config.encoderHiddenSize, numHeads: config.encoderNumHeads,
            numKVHeads: config.encoderNumKVHeads, headDim: config.headDim,
            rmsNormEps: config.rmsNormEps, ropeTheta: config.ropeTheta
        )
        postAttentionLayernorm = RMSNorm(dimensions: config.encoderHiddenSize, eps: config.rmsNormEps)
        mlp                    = SwiGLUMLP(hiddenSize: config.encoderHiddenSize,
                                           intermediateSize: config.encoderIntermediateSize)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, offset: Int = 0) -> MLXArray {
        let h = x + selfAttn(inputLayernorm(x), mask: mask, offset: offset)
        return h + mlp(postAttentionLayernorm(h))
    }
}

// MARK: - DiT Layer (AdaLN-Zero: self-attn + cross-attn + MLP)

final class AceStepDiTLayer: Module, @unchecked Sendable {
    let selfAttnNorm: RMSNorm
    let selfAttn: AceStepAttention
    let crossAttnNorm: RMSNorm
    let crossAttn: AceStepAttention
    let mlpNorm: RMSNorm
    let mlp: SwiGLUMLP
    var scaleShiftTable: MLXArray   // [1, 6, hiddenSize] — learned per-layer base

    init(config: AceStepConfig) {
        selfAttnNorm  = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        selfAttn      = AceStepAttention(
            hiddenSize: config.hiddenSize, numHeads: config.numHeads,
            numKVHeads: config.numKVHeads, headDim: config.headDim,
            rmsNormEps: config.rmsNormEps, ropeTheta: config.ropeTheta
        )
        crossAttnNorm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        crossAttn     = AceStepAttention(
            hiddenSize: config.hiddenSize, numHeads: config.numHeads,
            numKVHeads: config.numKVHeads, headDim: config.headDim,
            rmsNormEps: config.rmsNormEps, ropeTheta: config.ropeTheta,
            isCrossAttention: true
        )
        mlpNorm       = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        mlp           = SwiGLUMLP(hiddenSize: config.hiddenSize,
                                  intermediateSize: config.intermediateSize)
        scaleShiftTable = MLXArray.zeros([1, 6, config.hiddenSize])
        super.init()
    }

    // timestepProj: [B, 6, hiddenSize] from TimestepEmbedder
    // selfMask:    additive `[1,1,L,L]` or `[B,1,L,L]` for self-attn (sliding-window or none).
    // encoderMask: additive `[B,1,1,S]` for cross-attn (key padding).
    func callAsFunction(
        _ x: MLXArray,
        timestepProj: MLXArray,
        encoderHidden: MLXArray,
        selfMask: MLXArray? = nil,
        encoderMask: MLXArray? = nil,
        offset: Int = 0
    ) -> MLXArray {
        let B   = x.shape[0]
        let dim = x.shape[2]

        // AdaLN-Zero: scaleShiftTable [1,6,D] + timestepProj [B,6,D] → [B,6,D]
        let mod       = scaleShiftTable + timestepProj
        let shiftMSA  = mod[0..., 0, 0...].reshaped([B, 1, dim])
        let scaleMSA  = mod[0..., 1, 0...].reshaped([B, 1, dim])
        let gateMSA   = mod[0..., 2, 0...].reshaped([B, 1, dim])
        let shiftMLP  = mod[0..., 3, 0...].reshaped([B, 1, dim])
        let scaleMLP  = mod[0..., 4, 0...].reshaped([B, 1, dim])
        let gateMLP   = mod[0..., 5, 0...].reshaped([B, 1, dim])

        let n1 = selfAttnNorm(x) * (1 + scaleMSA) + shiftMSA
        var h  = x + gateMSA * selfAttn(n1, mask: selfMask, offset: offset)

        h = h + crossAttn(crossAttnNorm(h), encoderHidden: encoderHidden, mask: encoderMask)

        let n3 = mlpNorm(h) * (1 + scaleMLP) + shiftMLP
        return h + gateMLP * mlp(n3)
    }
}

// MARK: - Lyric Encoder
// Takes pre-computed Qwen3-Embedding outputs (1024-dim) → 2048-dim encoder states.
// Weight key prefix in checkpoint: encoder.lyric_encoder.*

final class AceLyricEncoder: Module, @unchecked Sendable {
    let embedTokens: Linear         // Linear(textHiddenDim=1024, hiddenSize=2048)
    let layers: [AceStepEncoderLayer]
    let norm: RMSNorm
    let config: AceStepConfig

    init(config: AceStepConfig) {
        self.config = config
        embedTokens = Linear(config.textHiddenDim, config.encoderHiddenSize, bias: true)
        layers      = (0..<config.numLyricEncoderLayers).map { _ in AceStepEncoderLayer(config: config) }
        norm        = RMSNorm(dimensions: config.encoderHiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    // embeds: [B, S, textHiddenDim] → [B, S, hiddenSize]
    //
    // NOTE: residuals across the 8 transformer layers compound to magnitudes around
    // ±3000 by layer 6 (verified against upstream Python). The final RMSNorm
    // normalizes those down to ±10, but in fp16 the per-layer MLP/attn intermediates
    // can hit Inf (max ≈ 65504) before reaching that final norm — softmax/RMSNorm
    // then propagate NaN through the rest of the pipeline. Run the encoder body in
    // fp32 to keep accumulation stable; cast back to the input dtype at the end.
    //
    // Layers alternate `sliding_attention` / `full_attention` per `attentionType(for:)` —
    // sliding layers get a `[1,1,L,L]` window mask, full layers get `nil`.
    func callAsFunction(_ embeds: MLXArray) -> MLXArray {
        let origDtype = embeds.dtype
        var h = embedTokens(embeds).asType(.float32)
        let L = h.shape[1]
        let slidingMask = config.useSlidingWindow
            ? slidingWindowMask(seqLen: L, window: config.slidingWindow)
            : nil
        for (idx, layer) in layers.enumerated() {
            let mask: MLXArray? = config.attentionType(for: idx) == "sliding_attention"
                ? slidingMask
                : nil
            h = layer(h, mask: mask)
        }
        return norm(h).asType(origDtype)
    }
}

// MARK: - Timbre Encoder
// Used for reference-audio (or silence-latent) timbre conditioning.
// Weight key prefix in checkpoint: encoder.timbre_encoder.*
//
// Upstream forward:
//   * `embed_tokens(refer_audio_acoustic_packed [N, T, 64]) → [N, T, 2048]`
//   * 4 transformer layers (`AceStepEncoderLayer`)
//   * final RMSNorm
//   * take position-0 hidden state per packed sample → [N, 2048]
//   * `unpack_timbre_embeddings(...)` → `[B, max_count, 2048]` + mask `[B, max_count]`
//
// The `special_token` parameter exists in the checkpoint but the upstream's
// forward path leaves it commented out (it would prepend a CLS-like token);
// position 0 is the first audio frame. We mirror that exactly.

final class AceStepTimbreEncoder: Module, @unchecked Sendable {
    let embedTokens: Linear            // Linear(timbreHiddenDim=64, hiddenSize=2048)
    var specialToken: MLXArray         // [1, 1, hiddenSize] — loaded but unused, matches upstream
    let layers: [AceStepEncoderLayer]
    let norm: RMSNorm
    let config: AceStepConfig

    init(config: AceStepConfig) {
        self.config  = config
        embedTokens  = Linear(config.timbreHiddenDim, config.encoderHiddenSize, bias: true)
        specialToken = MLXArray.zeros([1, 1, config.encoderHiddenSize])
        layers       = (0..<config.numTimbreEncoderLayers).map { _ in AceStepEncoderLayer(config: config) }
        norm         = RMSNorm(dimensions: config.encoderHiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    /// Run timbre encoder on packed reference-audio latents.
    ///
    /// - referAudioPacked: `[N, T, timbreHiddenDim]` — for text2music with B=1,
    ///   upstream uses `silence_latent[:, :750, :]` so N=1, T=750, dim=64.
    /// - returns: pooled timbre embedding of shape `[N, hiddenSize]` (position-0
    ///   hidden state). Wrap as `[B, count, hiddenSize]` at the call site for
    ///   `pack_sequences`. For B=1, N=1 the wrap is a single `reshape`.
    ///
    /// Layers alternate `sliding_attention` / `full_attention` (4 layers → s,f,s,f).
    func callAsFunction(_ referAudioPacked: MLXArray) -> MLXArray {
        let origDtype = referAudioPacked.dtype
        // fp32 internally for the same overflow reasons documented on the lyric encoder.
        var h = embedTokens(referAudioPacked).asType(.float32)
        let L = h.shape[1]
        let slidingMask = config.useSlidingWindow
            ? slidingWindowMask(seqLen: L, window: config.slidingWindow)
            : nil
        for (idx, layer) in layers.enumerated() {
            let mask: MLXArray? = config.attentionType(for: idx) == "sliding_attention"
                ? slidingMask
                : nil
            h = layer(h, mask: mask)
        }
        h = norm(h)
        // Take the first position per packed sample → [N, hiddenSize]
        let pooled = h[0..., 0, 0...]
        return pooled.asType(origDtype)
    }
}

// MARK: - Audio Detokenizer
// Converts quantized audio tokens (FSQ) → acoustic latents.
// Weight key prefix in checkpoint: detokenizer.*

final class AceAudioDetokenizer: Module, @unchecked Sendable {
    let embedTokens: Linear
    var specialTokens: MLXArray        // [1, poolWindowSize, hiddenSize]
    let layers: [AceStepEncoderLayer]
    let norm: RMSNorm
    let projOut: Linear

    init(config: AceStepConfig) {
        embedTokens   = Linear(config.encoderHiddenSize, config.encoderHiddenSize, bias: true)
        specialTokens = MLXArray.zeros([1, config.poolWindowSize, config.encoderHiddenSize])
        layers        = (0..<config.numDetokenizerLayers).map { _ in AceStepEncoderLayer(config: config) }
        norm          = RMSNorm(dimensions: config.encoderHiddenSize, eps: config.rmsNormEps)
        projOut       = Linear(config.encoderHiddenSize, config.audioAcousticHiddenDim, bias: true)
        super.init()
    }

    // x: [B, T_tok, hiddenSize] → [B, T_tok * P, audioAcousticHiddenDim]
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.shape[0]
        let T = x.shape[1]
        let P = specialTokens.shape[1]
        let D = specialTokens.shape[2]

        // Embed → [B, T, D], expand to [B, T, P, D]
        let emb  = embedTokens(x).reshaped([B, T, 1, D])
        let st   = specialTokens.reshaped([1, 1, P, D])
        var h    = (emb + st).reshaped([B * T, P, D])   // [B*T, P, D]

        for layer in layers { h = layer(h) }
        h = projOut(norm(h))                             // [B*T, P, audioAcousticHiddenDim]
        return h.reshaped([B, T * P, projOut.weight.shape[0]])
    }
}

// MARK: - DiT Model (decoder in checkpoint)

final class AceStepDiTModel: Module, @unchecked Sendable {
    let timeEmbed: TimestepEmbedder
    /// Residual timestep embedder. Used by ALL variants — Turbo passes
    /// `timestep_r ≠ timestep` for CFG distillation; SFT/base pass equal
    /// values so the input is 0, but the bias terms still contribute a
    /// non-zero, trained offset to `temb`.
    let timeEmbedR: TimestepEmbedder
    let conditionEmbedder: Linear
    let projIn: Conv1d
    let layers: [AceStepDiTLayer]
    let normOut: RMSNorm
    let projOut: ConvTransposed1d
    var scaleShiftTable: MLXArray      // [1, 2, hiddenSize]
    let patchSize: Int
    let config: AceStepConfig

    init(config: AceStepConfig) {
        self.config       = config
        patchSize         = config.patchSize
        timeEmbed         = TimestepEmbedder(freqDim: config.freqDim, hiddenSize: config.hiddenSize)
        timeEmbedR        = TimestepEmbedder(freqDim: config.freqDim, hiddenSize: config.hiddenSize)
        conditionEmbedder = Linear(config.encoderHiddenSize, config.hiddenSize, bias: true)
        projIn            = Conv1d(
            inputChannels:  config.inChannels,
            outputChannels: config.hiddenSize,
            kernelSize:     config.patchSize,
            stride:         config.patchSize
        )
        layers            = (0..<config.numDiTLayers).map { _ in AceStepDiTLayer(config: config) }
        normOut           = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        projOut           = ConvTransposed1d(
            inputChannels:  config.hiddenSize,
            outputChannels: config.audioAcousticHiddenDim,
            kernelSize:     config.patchSize,
            stride:         config.patchSize
        )
        scaleShiftTable   = MLXArray.zeros([1, 2, config.hiddenSize])
        super.init()
    }

    // hiddenStates:        [B, T, audioAcousticHiddenDim]  — noisy latent
    // contextLatents:      [B, T, inChannels - audioAcousticHiddenDim] — src + chunk mask
    // timestep/timestepR:  [B] float in [0,1]
    // encoderHiddenStates: [B, S, hiddenSize] — packed condition (lyric/timbre/text)
    // encoderAttentionMask:[B, S] int 0/1 — accepted for API compatibility but
    //   intentionally NOT applied to cross-attention. Upstream
    //   `modeling_acestep_v15_base.py:1391-1435` overwrites both `attention_mask`
    //   and `encoder_attention_mask` to `None` and rebuilds a fully-on 4D mask,
    //   so cross-attn attends to every encoder position (including post-padding
    //   values). The model was trained that way; masking padding here pulls
    //   conditioning off-distribution and produces audible artifacts.
    func callAsFunction(
        hiddenStates: MLXArray,
        contextLatents: MLXArray,
        timestep: MLXArray,
        timestepR: MLXArray,
        encoderHiddenStates: MLXArray,
        encoderAttentionMask: MLXArray? = nil
    ) -> MLXArray {
        _ = encoderAttentionMask  // upstream-parity: discarded, see comment above
        let B   = hiddenStates.shape[0]
        let dim = scaleShiftTable.shape[2]

        // Concatenate context + noisy audio → [B, T, inChannels]
        var x = concatenated([contextLatents, hiddenStates], axis: -1)
        let originalLen = x.shape[1]

        // Pad to patchSize multiple
        let rem = originalLen % patchSize
        if rem != 0 {
            let pad = patchSize - rem
            let zeros = MLXArray.zeros([B, pad, x.shape[2]])
            x = concatenated([x, zeros], axis: 1)
        }

        // Conv1d patch embed: [B, T, inChannels] → [B, T//patchSize, hiddenSize]
        x = projIn(x)

        // Project encoder hidden states
        let encH = conditionEmbedder(encoderHiddenStates)

        // Self-attn sliding mask is over the patched audio sequence (T_p = T // patchSize).
        // Cross-attn uses no mask — see signature comment.
        let Tp = x.shape[1]
        let slidingSelfMask = (config.useSlidingWindow && Tp > 1)
            ? slidingWindowMask(seqLen: Tp, window: config.slidingWindow)
            : nil
        let crossMask: MLXArray? = nil

        // Timestep embeddings: temb [B, H], proj [B, 6, H].
        // Upstream calls both embedders for ALL variants. For SFT/base, `timestep_r`
        // equals `timestep` so the residual input is 0, but `time_embed_r(0)` is a
        // non-zero, trained constant offset (Linear biases). Skipping it on SFT/base
        // produces out-of-distribution time conditioning → garbled audio.
        let (tembT, projT) = timeEmbed(timestep)
        let (tembR, projR) = timeEmbedR(timestep - timestepR)
        let temb         = tembT + tembR
        let timestepProj = projT + projR

        for (idx, layer) in layers.enumerated() {
            let selfMask: MLXArray? = config.attentionType(for: idx) == "sliding_attention"
                ? slidingSelfMask
                : nil
            x = layer(
                x,
                timestepProj: timestepProj,
                encoderHidden: encH,
                selfMask: selfMask,
                encoderMask: crossMask
            )
        }

        // AdaLN output norm
        let mod   = scaleShiftTable + temb.reshaped([B, 1, dim])
        let shift = mod[0..., 0, 0...].reshaped([B, 1, dim])
        let scale = mod[0..., 1, 0...].reshaped([B, 1, dim])
        x = normOut(x) * (1 + scale) + shift

        // ConvTransposed1d depatch: [B, T//2, hiddenSize] → [B, T, audioAcousticHiddenDim]
        x = projOut(x)

        // Crop to original length
        return x[0..., ..<originalLen, 0...]
    }
}

// MARK: - Top-level model (AceStepConditionGenerationModel in checkpoint)

final class ACEStepDiT: Module, @unchecked Sendable {
    let decoder: AceStepDiTModel
    let lyricEncoder: AceLyricEncoder
    let timbreEncoder: AceStepTimbreEncoder
    let detokenizer: AceAudioDetokenizer
    /// Audio tokenizer (input projection + attention pooler + ResidualFSQ).
    /// Used by cover/text2musicLM modes to convert 25 Hz acoustic latents into
    /// 5 Hz quantized tokens (and back through `detokenizer`).
    let audioTokenizer: AceStepAudioTokenizer
    /// Projects external text-encoder hidden states (1024) → DiT hidden size (2048).
    /// Loaded from checkpoint key `encoder.text_projector.weight` (no bias).
    let textProjector: Linear
    var nullConditionEmb: MLXArray   // [1, 1, hiddenSize]
    let config: AceStepConfig

    init(config: AceStepConfig = .turbo) {
        self.config    = config
        decoder        = AceStepDiTModel(config: config)
        lyricEncoder   = AceLyricEncoder(config: config)
        timbreEncoder  = AceStepTimbreEncoder(config: config)
        detokenizer    = AceAudioDetokenizer(config: config)
        audioTokenizer = AceStepAudioTokenizer(config: config)
        textProjector  = Linear(config.textHiddenDim, config.encoderHiddenSize, bias: false)
        nullConditionEmb = MLXArray.zeros([1, 1, config.encoderHiddenSize])
        super.init()
    }
}

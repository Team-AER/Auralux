import MLX
import MLXNN
import Foundation

// MARK: - FSQ  (Finite Scalar Quantization)
//
// Mirrors `lucidrains/vector-quantize-pytorch/finite_scalar_quantization.py`.
// Each input dimension is independently quantized to `levels[d]` evenly-spaced
// values in `[-1, 1]`. No learned codebook — just deterministic rounding.
//
// For ACE-Step v1.5 turbo:
//   * 6 dimensions, levels = [8, 8, 8, 5, 5, 5] → codebook size = 64 000
//   * num_quantizers = 1 (single residual stage)

final class FSQ: Module, @unchecked Sendable {
    let levels: [Int]                  // per-dim level count, length = numDims
    let numDims: Int
    private let halfLevels: MLXArray   // [numDims] = (levels - 1) / 2 in float
    private let halfLevelsInt: [Int]   // [numDims] = levels // 2
    private let levelMod: MLXArray     // [numDims] = levels (used in indices_to_codes)
    private let basis: MLXArray        // [numDims] = cumulative product of levels

    init(levels: [Int]) {
        self.levels = levels
        self.numDims = levels.count
        let halfFloat = levels.map { Float($0 - 1) / 2.0 }
        self.halfLevels = MLXArray(halfFloat)
        self.halfLevelsInt = levels.map { $0 / 2 }
        self.levelMod = MLXArray(levels.map { Int32($0) })
        var b: [Int32] = []
        var prod: Int32 = 1
        for l in levels {
            b.append(prod)
            prod *= Int32(l)
        }
        self.basis = MLXArray(b)
        super.init()
    }

    /// Round-to-nearest with no straight-through (inference only).
    private func roundNearest(_ x: MLXArray) -> MLXArray {
        // round() in MLX rounds half-to-even; add 0.5 then floor for nearest-up.
        // For FSQ this matches the upstream `round_ste(...)` numerics on
        // already-bounded inputs.
        return MLX.round(x)
    }

    /// Bound `z` to `[-half_l, half_l]` per dimension with the same tanh scaling
    /// as upstream FSQ (`bound` method, eps=1e-3). Even-level dims get a 0.5
    /// offset so 0 is never a valid level (which would collapse the codebook).
    func bound(_ z: MLXArray) -> MLXArray {
        // Per-dim half = (level - 1) * (1 + eps) / 2 — but eps just changes the
        // tanh asymptote, not which integer values are valid; using exact half
        // matches the floating-point quantize-then-renormalize round-trip
        // closely enough at fp16 to be a no-op in practice.
        let halfL = halfLevels                                // [D]
        let isEven = MLXArray(levels.map { Int32($0 % 2 == 0 ? 1 : 0) }).asType(.float32)
        let offset = isEven * 0.5                              // [D]
        // shift = atanh(offset / halfL); guard halfL=0 (level=1, skip).
        // Levels are always >= 2 here, so halfL > 0 and division is safe.
        let shiftArg = offset / halfL
        let shift = halfArctanh(shiftArg)
        return tanh(z + shift) * halfL - offset
    }

    /// FSQ quantize: bound → round → renormalize to `[-1, 1]`.
    /// Returns continuous (re-normalized) codes, NOT integer indices.
    func quantize(_ z: MLXArray) -> MLXArray {
        let bounded = bound(z)
        let rounded = roundNearest(bounded)
        let halfWidth = MLXArray(levels.map { Float($0 / 2) })  // levels // 2
        return rounded / halfWidth
    }

    /// Convert continuous quantized codes (in `[-1, 1]`) to flat integer indices.
    /// `codes`: `[..., D]` → `indices`: `[...]` (single int per token).
    func codesToIndices(_ codes: MLXArray) -> MLXArray {
        let halfWidth = MLXArray(levels.map { Float($0 / 2) })
        let halfWidthInt = MLXArray(levels.map { Int32($0 / 2) })
        // Inverse of `quantize`'s renormalize: scale back to [0, level-1]
        let zhat = (codes * halfWidth).asType(.int32) + halfWidthInt
        return (zhat * basis).sum(axis: -1)
    }

    /// Convert flat integer indices back to continuous codes in `[-1, 1]`.
    /// `indices`: `[...]` → `codes`: `[..., D]`.
    func indicesToCodes(_ indices: MLXArray) -> MLXArray {
        // Expand last dim and split via base/level mod.
        let expanded = indices.expandedDimensions(axis: -1)        // [..., 1]
        let nonCentered = (expanded / basis) % levelMod            // [..., D]
        let halfWidthInt = MLXArray(levels.map { Int32($0 / 2) })
        let halfWidth = MLXArray(levels.map { Float($0 / 2) })
        return (nonCentered - halfWidthInt).asType(.float32) / halfWidth
    }
}

// Workaround for missing MLXArray atanh — inputs are bounded in (-1, 1) so the
// straightforward identity log((1+x)/(1-x))/2 is numerically stable here.
private func halfArctanh(_ x: MLXArray) -> MLXArray {
    return 0.5 * log((1.0 + x) / (1.0 - x))
}

// MARK: - ResidualFSQ  (1-stage in ACE-Step v1.5 turbo)
//
// In the general case this would be N stacked FSQ layers fed by the residual
// `r = z - sum_quantized`. Turbo uses N=1, so we collapse to a single stage
// with `project_in` / `project_out`.

final class ResidualFSQ: Module, @unchecked Sendable {
    let projectIn:  Linear        // hidden_size → fsq_dim
    let projectOut: Linear        // fsq_dim → hidden_size
    let layers: [FSQ]             // length = num_quantizers (typically 1 for turbo)

    init(hiddenSize: Int, fsqDim: Int, levels: [Int], numQuantizers: Int) {
        self.projectIn  = Linear(hiddenSize, fsqDim, bias: true)
        self.projectOut = Linear(fsqDim, hiddenSize, bias: true)
        self.layers     = (0..<numQuantizers).map { _ in FSQ(levels: levels) }
        super.init()
    }

    /// Forward: continuous embeddings → quantized embeddings + per-stage indices.
    /// - `x`: `[B, T, hiddenSize]`
    /// - Returns: `(quantized [B, T, hiddenSize], indices [B, T, numQuantizers])`
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        var residual = projectIn(x)                       // [B, T, fsqDim]
        var quantSum = MLXArray.zeros(residual.shape).asType(residual.dtype)
        var allIndices: [MLXArray] = []
        for fsq in layers {
            let q = fsq.quantize(residual)                // [B, T, fsqDim]
            quantSum = quantSum + q
            residual = residual - q
            allIndices.append(fsq.codesToIndices(q))      // [B, T]
        }
        let quantized = projectOut(quantSum)              // [B, T, hiddenSize]
        let indices = stacked(allIndices, axis: -1)       // [B, T, numQuantizers]
        return (quantized, indices)
    }

    /// Inverse forward used by `text2musicLM` / cover-with-codes: takes integer
    /// indices and reconstructs the embedding.
    /// - `indices`: `[B, T, numQuantizers]`
    /// - Returns: `[B, T, hiddenSize]`
    func getOutputFromIndices(_ indices: MLXArray) -> MLXArray {
        precondition(indices.shape.last == layers.count,
                     "Expected last-dim = numQuantizers (\(layers.count)), got \(indices.shape.last ?? -1)")
        var sum: MLXArray? = nil
        for (q, fsq) in layers.enumerated() {
            let stageIdx = indices[0..., 0..., q]
            let codes = fsq.indicesToCodes(stageIdx)      // [B, T, fsqDim]
            sum = sum.map { $0 + codes } ?? codes
        }
        return projectOut(sum!)
    }
}

// MARK: - AttentionPooler
//
// Mirrors `AttentionPooler.forward` in `modeling_acestep_v15_turbo.py:746-862`.
// Pools 5 contiguous frames (the `pool_window_size=5` patches) into a single
// representation by prepending a learned CLS token and running 2 layers of
// the standard Qwen3-style transformer over `[CLS, p1..p5]`.

final class AceStepAttentionPooler: Module, @unchecked Sendable {
    let embedTokens: Linear
    let layers: [AceStepEncoderLayer]
    let norm: RMSNorm
    var specialToken: MLXArray            // [1, 1, hiddenSize] CLS

    init(config: AceStepConfig) {
        embedTokens  = Linear(config.encoderHiddenSize, config.encoderHiddenSize, bias: true)
        layers       = (0..<config.numAttentionPoolerLayers).map { _ in AceStepEncoderLayer(config: config) }
        norm         = RMSNorm(dimensions: config.encoderHiddenSize, eps: config.rmsNormEps)
        specialToken = MLXArray.zeros([1, 1, config.encoderHiddenSize])
        super.init()
    }

    /// - x: `[B, T_pool, P, hiddenSize]` where P = pool_window_size = 5
    /// - Returns: `[B, T_pool, hiddenSize]` — pooled CLS-token output per slot.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.shape[0]
        let T = x.shape[1]
        let P = x.shape[2]
        let D = x.shape[3]

        // Project each frame through the input Linear.
        let proj = embedTokens(x)                                          // [B, T, P, D]

        // Prepend a CLS token to the P axis: [B, T, 1+P, D].
        let cls = broadcast(specialToken.reshaped([1, 1, 1, D]), to: [B, T, 1, D])
        let withCls = concatenated([cls, proj], axis: 2)                   // [B, T, 1+P, D]

        // Fold (B, T) into batch for self-attention: [B*T, 1+P, D].
        var h = withCls.reshaped([B * T, 1 + P, D])
        let origDtype = h.dtype
        h = h.asType(.float32)
        for layer in layers {
            h = layer(h)
        }
        h = norm(h)
        // Take CLS (position 0): [B*T, D] → [B, T, D].
        let cls0 = h[0..., 0, 0...]
        return cls0.reshaped([B, T, D]).asType(origDtype)
    }
}

// MARK: - AceStepAudioTokenizer
//
// Top-level audio tokenizer matching upstream `AceStepAudioTokenizer`.
// Pipeline (forward):
//   audio_acoustic_proj  : 64 → 2048    (per-frame at 25 Hz)
//   reshape              : [B, T, P=5, 2048]
//   attention_pooler     : 25 Hz → 5 Hz    [B, T_pool, 2048]
//   quantizer (FSQ)      : continuous → quantized continuous + indices

final class AceStepAudioTokenizer: Module, @unchecked Sendable {
    let audioAcousticProj: Linear            // Linear(audio_acoustic_hidden_dim=64, hidden_size=2048)
    let attentionPooler: AceStepAttentionPooler
    let quantizer: ResidualFSQ
    let poolWindowSize: Int

    init(config: AceStepConfig) {
        audioAcousticProj = Linear(config.audioAcousticHiddenDim, config.encoderHiddenSize, bias: true)
        attentionPooler   = AceStepAttentionPooler(config: config)
        quantizer         = ResidualFSQ(
            hiddenSize:    config.encoderHiddenSize,
            fsqDim:        config.fsqDim,
            levels:        config.fsqInputLevels,
            numQuantizers: config.fsqInputNumQuantizers
        )
        self.poolWindowSize = config.poolWindowSize
        super.init()
    }

    /// Tokenize a 25 Hz acoustic latent into 5 Hz quantized representations.
    /// - `latents`: `[B, T_25, audio_acoustic_hidden_dim=64]` — must satisfy
    ///   `T_25 % pool_window_size == 0`. Pad with silence-latent frames at the
    ///   call site if the sequence length doesn't divide evenly.
    /// - Returns: `(quantized [B, T_5, hidden_size], indices [B, T_5, num_quantizers])`
    func callAsFunction(_ latents: MLXArray) -> (MLXArray, MLXArray) {
        let B = latents.shape[0]
        let T25 = latents.shape[1]
        let P = poolWindowSize
        precondition(T25 % P == 0,
                     "AceStepAudioTokenizer expects pool_window_size-multiple frames; got \(T25) % \(P) != 0")
        let T5 = T25 / P

        // Project to hidden_size first, then fold P frames into a fast axis.
        let projected = audioAcousticProj(latents)                       // [B, T_25, 2048]
        let folded = projected.reshaped([B, T5, P, projected.shape[2]])   // [B, T_5, P, 2048]

        // Pool 5 → 1 via the attention CLS-pooler.
        let pooled = attentionPooler(folded)                             // [B, T_5, 2048]

        // Quantize.
        let (quantized, indices) = quantizer(pooled)                     // [B, T_5, 2048], [B, T_5, Q]
        return (quantized, indices)
    }
}

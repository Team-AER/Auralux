import MLX
import MLXNN
import Foundation

// MARK: - Turbo timestep schedule
//
// Mirrors `acestep/inference.py`:
//   raw = [1.0 - i / n for i in range(n)]
//   if shift != 1.0:
//       raw = [shift * t / (1.0 + (shift - 1.0) * t) for t in raw]
//
// Shift transforms:
//   * shift=1.0 → identity (production default per acestep/inference.py:136).
//   * shift=2.0/3.0 compress the late-denoising steps and shift mass toward
//     the start. Empirically shift=3.0 produces ~15-20 % quieter audio.
//
// Upstream caps `infer_steps` at 20.

enum FlowScheduleError: Error, CustomStringConvertible {
    case invalidNumSteps(Int, max: Int)
    case invalidShift(Double)

    var description: String {
        switch self {
        case .invalidNumSteps(let n, let max):
            return "numSteps must be in 1...\(max) (got \(n))"
        case .invalidShift(let s):
            return "scheduleShift must be one of {1.0, 2.0, 3.0} (got \(s))"
        }
    }
}

// Keep old error name as a typealias so any existing catch sites still compile.
typealias TurboScheduleError = FlowScheduleError

/// Build a flow-matching denoising schedule of length `numSteps`. Returns
/// timesteps in descending order, all in (0, 1]. Final step `0` is implicit.
func buildFlowSchedule(numSteps: Int, maxSteps: Int, shift: Double) throws -> [Float] {
    guard (1...maxSteps).contains(numSteps) else {
        throw FlowScheduleError.invalidNumSteps(numSteps, max: maxSteps)
    }
    let validShifts: Set<Double> = [1.0, 2.0, 3.0]
    guard validShifts.contains(shift) else {
        throw FlowScheduleError.invalidShift(shift)
    }
    let raw = (0..<numSteps).map { i in 1.0 - Double(i) / Double(numSteps) }
    if shift == 1.0 { return raw.map(Float.init) }
    return raw.map { t in Float(shift * t / (1.0 + (shift - 1.0) * t)) }
}

func buildTurboSchedule(numSteps: Int, shift: Double) throws -> [Float] {
    try buildFlowSchedule(numSteps: numSteps, maxSteps: 20, shift: shift)
}

// MARK: - TurboSampler

/// Default 8-step shift=1.0 schedule. Pre-computed because the throwing
/// builder is overkill for this hot-path and `init()` shouldn't throw.
private let kDefaultTurboSchedule: [Float] = [
    1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125
]

/// N-step ODE (Euler) sampler for ACE-Step v1.5 Turbo.
///
/// Flow-matching prediction: the model outputs a velocity field v(x_t, t).
/// ODE update:  x_{t+1} = x_t − v * dt
/// Final step:  x_0     = x_t − v * t   (i.e. `get_x0_from_noise`)
struct TurboSampler {
    let schedule: [Float]

    /// Inputs for repaint mode. The sampler keeps non-mask frames pinned to
    /// the user's source audio (re-noised to match `t_next`) for the first
    /// `injectionRatio * numSteps` steps, and at the end blends the boundary
    /// frames back to the source via a linear cross-fade of `crossfadeFrames`
    /// frames on either side. Mirrors `_repaint_step_injection` and
    /// `_repaint_boundary_blend` from `modeling_acestep_v15_turbo.py:1561-1590`.
    struct RepaintInputs {
        let cleanSrcLatents: MLXArray   // [1, T, 64]
        let mask: MLXArray              // [1, T] int32 0/1  (1 = repaint here)
        let injectionRatio: Double      // ∈ [0, 1]; upstream default 0.5
        let crossfadeFrames: Int        // ≥ 0; upstream default 10
        let noise: MLXArray             // shared noise tensor for renoise
    }

    init(schedule: [Float] = kDefaultTurboSchedule) {
        self.schedule = schedule
    }

    /// Validating initializer that throws on out-of-range numSteps or a
    /// shift not in {1.0, 2.0, 3.0}. Use this when the caller is exposing
    /// these values to user input.
    init(numSteps: Int, shift: Double) throws {
        self.schedule = try buildTurboSchedule(numSteps: numSteps, shift: shift)
    }

    var numSteps: Int { schedule.count }

    /// Run the full denoising loop.
    ///
    /// - Parameters:
    ///   - noise:                 [B, T, audioAcousticHiddenDim] initial Gaussian noise
    ///   - contextLatents:        [B, T, 128] context (silence/src latents + chunk masks)
    ///   - encoderHiddenStates:   [B, S, hiddenSize] from lyric/condition encoder
    ///   - encoderAttentionMask:  [B, S] int 0/1 packed pad mask, or `nil` (no padding)
    ///   - model:                 The DiT decoder
    ///   - repaint:               Optional repaint params (see `RepaintInputs`).
    ///   - onStep:                Progress callback (step index, total steps)
    /// - Returns: Denoised acoustic latent [B, T, audioAcousticHiddenDim]
    func sample(
        noise: MLXArray,
        contextLatents: MLXArray,
        encoderHiddenStates: MLXArray,
        encoderAttentionMask: MLXArray? = nil,
        model: AceStepDiTModel,
        repaint: RepaintInputs? = nil,
        onStep: ((Int, Int) -> Void)? = nil
    ) throws -> MLXArray {
        let B  = noise.shape[0]
        var xt = noise

        let injectionCutoff: Int = {
            guard let repaint else { return 0 }
            return Int((Double(schedule.count) * repaint.injectionRatio).rounded())
        }()

        for (i, t) in schedule.enumerated() {
            try Task.checkCancellation()
            let tTensor = MLXArray(Array(repeating: t, count: B))

            let vt = model.callAsFunction(
                hiddenStates:         xt,
                contextLatents:       contextLatents,
                timestep:             tTensor,
                timestepR:            tTensor,
                encoderHiddenStates:  encoderHiddenStates,
                encoderAttentionMask: encoderAttentionMask
            )

            let tAfter: Float
            if i == schedule.count - 1 {
                // get_x0_from_noise: x0 = xt - vt * t
                xt = xt - vt * MLXArray(t)
                tAfter = 0
            } else {
                let dt = t - schedule[i + 1]
                xt = xt - vt * MLXArray(dt)
                tAfter = schedule[i + 1]
            }

            // Repaint step injection: replace non-mask regions of `xt` with
            // re-noised source latents. Only fires for the first
            // `injection_ratio * numSteps` steps.
            if let repaint, i < injectionCutoff {
                xt = repaintStepInjection(
                    xt:        xt,
                    cleanSrc:  repaint.cleanSrcLatents,
                    mask:      repaint.mask,
                    tNext:     tAfter,
                    noise:     repaint.noise
                )
            }

            eval(xt)
            onStep?(i, numSteps)
        }

        // Final cross-fade blend at repaint boundaries (smooths the jump
        // between source-passthrough frames and freshly-generated frames).
        if let repaint, repaint.crossfadeFrames > 0 {
            xt = repaintBoundaryBlend(
                xGen:      xt,
                cleanSrc:  repaint.cleanSrcLatents,
                mask:      repaint.mask,
                cfFrames:  repaint.crossfadeFrames
            )
            eval(xt)
        }

        return xt
    }
}

// MARK: - Repaint helpers
//
// Direct port of upstream `_repaint_step_injection` and
// `_repaint_boundary_blend` from `modeling_acestep_v15_turbo.py:1561-1590`.

/// `zt = t_next * noise + (1 - t_next) * clean_src`  for non-mask frames;
/// keep `xt` where `mask = 1` (repaint region).
private func repaintStepInjection(
    xt: MLXArray, cleanSrc: MLXArray, mask: MLXArray, tNext: Float, noise: MLXArray
) -> MLXArray {
    let zt = MLXArray(tNext) * noise + (1.0 - MLXArray(tNext)) * cleanSrc
    // mask: [1, T] → [1, T, 1] for broadcast over feature dim.
    let m = mask.expandedDimensions(axis: -1).asType(xt.dtype)
    return m * xt + (1.0 - m) * zt
}

/// Per-batch boundary feathering: linear ramps of length `cfFrames` on either
/// side of the repaint region, clamped to the sequence boundaries. Produces a
/// soft mask in `[0, 1]` and blends `mask*x_gen + (1-mask)*clean_src`.
private func repaintBoundaryBlend(
    xGen: MLXArray, cleanSrc: MLXArray, mask: MLXArray, cfFrames: Int
) -> MLXArray {
    let B = mask.shape[0]
    let T = mask.shape[1]
    eval(mask)
    let bits = mask.asArray(Int32.self)
    var soft = [Float](repeating: 0, count: B * T)
    for b in 0..<B {
        // Find contiguous runs of mask=1; precompute run boundaries [left, right).
        var runs: [(Int, Int)] = []
        var i = 0
        while i < T {
            if bits[b * T + i] == 1 {
                var j = i
                while j < T && bits[b * T + j] == 1 { j += 1 }
                runs.append((i, j))
                i = j
            } else {
                i += 1
            }
        }
        // Bake hard mask first.
        for (l, r) in runs {
            for k in l..<r { soft[b * T + k] = 1.0 }
        }
        // Add fades. Two adjacent runs share their inner boundary's mask;
        // we just add the fade on the outside of each run, clamped at T/0.
        for (l, r) in runs {
            let fs = max(l - cfFrames, 0)
            let leftLen = l - fs
            if leftLen > 0 {
                for k in 0..<leftLen {
                    // Linear ramp 0 → 1 across leftLen+2 endpoints, exclude both.
                    let v = Float(k + 1) / Float(leftLen + 1)
                    soft[b * T + fs + k] = max(soft[b * T + fs + k], v)
                }
            }
            let fe = min(r + cfFrames, T)
            let rightLen = fe - r
            if rightLen > 0 {
                for k in 0..<rightLen {
                    let v = Float(rightLen - k) / Float(rightLen + 1)
                    soft[b * T + r + k] = max(soft[b * T + r + k], v)
                }
            }
        }
    }
    let softMask = MLXArray(soft, [B, T, 1]).asType(xGen.dtype)
    return softMask * xGen + (1.0 - softMask) * cleanSrc
}

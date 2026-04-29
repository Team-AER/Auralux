import MLX
import MLXNN
import Foundation

// MARK: - CFGSampler

/// N-step Euler sampler for ACE-Step v1.5 SFT / Base.
///
/// Uses APG (Adaptive Prompt Guidance) — the default guidance method in
/// `modeling_acestep_v15_base.py` (`apg_forward`, norm_threshold=2.5, eta=0).
/// Simple linear CFG at scale=15 diverges because the velocity diff is large;
/// APG clips the per-element RMS of diff to ≤ 2.5, then projects orthogonal
/// to v_cond. A momentum buffer (momentum=-0.75) smooths diff across steps.
///
/// ODE update:  x_{t+1} = x_t − v_guided * dt
/// Final step:  x_0     = x_t − v_guided * t
struct CFGSampler {
    let schedule: [Float]
    let cfgScale: Float
    // guidanceInterval ∈ (0, 1]: fraction of steps where APG is active (middle portion).
    // Python pipeline default: 0.5 — guidance only on the central 50% of steps.
    // start = floor(N * (1 - interval) / 2),  end = floor(N * (interval/2 + 0.5))
    let guidanceInterval: Float

    init(schedule: [Float], cfgScale: Float, guidanceInterval: Float = 0.5) {
        self.schedule         = schedule
        self.cfgScale         = cfgScale
        self.guidanceInterval = guidanceInterval
    }

    init(numSteps: Int, shift: Double, cfgScale: Float, guidanceInterval: Float = 0.5) throws {
        self.schedule         = try buildFlowSchedule(numSteps: numSteps, maxSteps: 100, shift: shift)
        self.cfgScale         = cfgScale
        self.guidanceInterval = guidanceInterval
    }

    var numSteps: Int { schedule.count }

    func sample(
        noise: MLXArray,
        contextLatents: MLXArray,
        encoderHiddenStates: MLXArray,
        encoderAttentionMask: MLXArray? = nil,
        nullConditionEmb: MLXArray,
        model: AceStepDiTModel,
        onStep: ((Int, Int) -> Void)? = nil
    ) -> MLXArray {
        let B  = noise.shape[0]
        let S  = encoderHiddenStates.shape[1]
        let N  = schedule.count
        var xt = noise

        let startIdx = Int(Float(N) * (1.0 - guidanceInterval) / 2.0)
        let endIdx   = Int(Float(N) * (guidanceInterval / 2.0 + 0.5))

        // Tile null_condition_emb [1, 1, H] → [B, S, H].
        // Python v1.5: null_condition_emb.expand_as(encoder_hidden_states) — same shape + mask.
        let nullEmb = tiled(nullConditionEmb, repetitions: [B, S, 1])

        // Momentum buffer persists across guidance steps (Python MomentumBuffer(momentum=-0.75)).
        var momentumRunning: MLXArray? = nil

        for (i, t) in schedule.enumerated() {
            let tTensor = MLXArray(Array(repeating: t, count: B))

            let vCond = model(
                hiddenStates:         xt,
                contextLatents:       contextLatents,
                timestep:             tTensor,
                timestepR:            tTensor,
                encoderHiddenStates:  encoderHiddenStates,
                encoderAttentionMask: encoderAttentionMask
            )

            let vt: MLXArray
            if i >= startIdx && i < endIdx {
                // Inside guidance window: APG (two passes)
                let vUncond = model(
                    hiddenStates:         xt,
                    contextLatents:       contextLatents,
                    timestep:             tTensor,
                    timestepR:            tTensor,
                    encoderHiddenStates:  nullEmb,
                    encoderAttentionMask: encoderAttentionMask
                )
                vt = apgGuidance(
                    vCond: vCond,
                    vUncond: vUncond,
                    scale: cfgScale,
                    momentumRunning: &momentumRunning
                )
            } else {
                // Outside guidance window: conditional only (no unconditional pass)
                vt = vCond
            }

            if i == schedule.count - 1 {
                xt = xt - vt * MLXArray(t)
            } else {
                let dt = t - schedule[i + 1]
                xt = xt - vt * MLXArray(dt)
            }

            eval(xt)
            onStep?(i, numSteps)
        }

        return xt
    }
}

// MARK: - APG guidance

/// Adaptive Prompt Guidance — mirrors `apg_forward()` in `acestep/apg_guidance.py`.
///
/// 1. Momentum: running_avg = diff + (-0.75) * running_avg  (smooths diff across steps).
/// 2. Norm-clip: clips diff when per-element RMS > 2.5.
///    Python: diff.norm() / ones.norm() = L2 / sqrt(T*H) = RMS.  Must divide by sqrt(T*H).
/// 3. Orthogonal projection: removes component of diff parallel to v_cond (eta=0).
/// 4. Returns v_cond + (scale-1) * diff_orthogonal.
private func apgGuidance(
    vCond: MLXArray,
    vUncond: MLXArray,
    scale: Float,
    normThreshold: Float = 2.5,
    momentumRunning: inout MLXArray?
) -> MLXArray {
    var diff = vCond - vUncond

    // Momentum buffer: running = diff + momentum * running  (momentum = -0.75)
    if let running = momentumRunning {
        let newRunning = diff + MLXArray(Float(-0.75)) * running
        momentumRunning = newRunning
        diff = newRunning
    } else {
        // First guidance step: running_average starts at 0, so running = diff + 0 = diff
        momentumRunning = diff
    }

    // Per-element RMS norm: L2(diff) / sqrt(T*H), matching Python's diff.norm()/ones.norm()
    let T = Float(diff.shape[1])
    let H = Float(diff.shape[2])
    let diffNorm = sqrt((diff * diff).sum(axes: [1, 2], keepDims: true)) / MLXArray((T * H).squareRoot())
    // clipScale = min(1, threshold / rmsNorm) — identity when rms ≤ threshold
    let clipScale = MLXArray(normThreshold) / maximum(diffNorm, MLXArray(normThreshold))
    diff = diff * clipScale

    // Project diff orthogonal to v_cond (eta=0: discard parallel component entirely)
    let dot    = (diff * vCond).sum(axes: [1, 2], keepDims: true)
    let normSq = (vCond * vCond).sum(axes: [1, 2], keepDims: true) + Float(1e-8)
    let diffOrthogonal = diff - (dot / normSq) * vCond

    return vCond + MLXArray(scale - 1.0) * diffOrthogonal
}

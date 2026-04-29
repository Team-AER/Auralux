import MLX
import MLXNN
import Foundation

// MARK: - CFGSampler

/// N-step Euler sampler for ACE-Step v1.5 SFT / Base.
///
/// Uses APG (Adaptive Prompt Guidance) — the default guidance method in
/// `modeling_acestep_v15_base.py` (`apg_forward`, norm_threshold=2.5, eta=0).
/// Simple linear CFG at scale=15 diverges because the velocity diff is large;
/// APG clips the per-feature L2 norm of diff (over time) to ≤ 2.5, then projects
/// orthogonal to v_cond per feature. Momentum buffer (−0.75) smooths diff across steps.
///
/// ODE update:  x_{t+1} = x_t − v_guided * dt
/// Final step:  x_0     = x_t − v_guided * t
struct CFGSampler {
    let schedule: [Float]
    let cfgScale: Float
    // guidanceInterval ∈ (0, 1]: fraction of steps where APG is active (middle portion).
    // Upstream default: 1.0 — CFG applied at every step (cfg_interval_start=0.0, end=1.0).
    // start = floor(N * (1 - interval) / 2),  end = floor(N * (interval/2 + 0.5))
    let guidanceInterval: Float

    init(schedule: [Float], cfgScale: Float, guidanceInterval: Float = 1.0) {
        self.schedule         = schedule
        self.cfgScale         = cfgScale
        self.guidanceInterval = guidanceInterval
    }

    init(numSteps: Int, shift: Double, cfgScale: Float, guidanceInterval: Float = 1.0) throws {
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
    ) throws -> MLXArray {
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
            try Task.checkCancellation()
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
/// 2. Norm-clip: for each feature h, clips L2 norm of diff[:, :, h] over time to ≤ 2.5.
///    Upstream: diff.norm(p=2, dim=[1], keepdim=True) — per-feature L2 over T, shape [B, 1, H].
/// 3. Orthogonal projection: removes component of diff parallel to v_cond (eta=0), per feature.
///    Upstream: project(diff, v_cond, dims=[1]) — projection along T for each H independently.
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

    // Per-feature L2 norm over time: shape [B, 1, H] — matches upstream dim=[1].
    let diffNorm = sqrt((diff * diff).sum(axes: [1], keepDims: true))
    // clipScale = min(1, threshold / norm) — identity when norm ≤ threshold
    let clipScale = MLXArray(normThreshold) / maximum(diffNorm, MLXArray(normThreshold))
    diff = diff * clipScale

    // Project diff orthogonal to v_cond per feature (eta=0: discard parallel component).
    // Matches upstream project(diff, v_cond, dims=[1]): dot product along T for each H.
    let dot    = (diff * vCond).sum(axes: [1], keepDims: true)           // [B, 1, H]
    let normSq = (vCond * vCond).sum(axes: [1], keepDims: true) + Float(1e-8)  // [B, 1, H]
    let diffOrthogonal = diff - (dot / normSq) * vCond

    return vCond + MLXArray(scale - 1.0) * diffOrthogonal
}

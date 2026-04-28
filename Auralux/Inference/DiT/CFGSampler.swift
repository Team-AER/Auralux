import MLX
import MLXNN
import Foundation

// MARK: - CFGSampler

/// N-step Euler sampler for ACE-Step v1.5 SFT / Base.
///
/// Unlike TurboSampler (one pass, CFG distilled), SFT uses standard
/// classifier-free guidance: two DiT forward passes per step, blended by
/// `cfgScale`.  mirrors `modeling_acestep_v15.py` (non-turbo inference path).
///
/// ODE update:  x_{t+1} = x_t − v_cfg * dt
/// Final step:  x_0     = x_t − v_cfg * t
/// where v_cfg = v_uncond + cfgScale * (v_cond - v_uncond)
struct CFGSampler {
    let schedule: [Float]
    let cfgScale: Float

    init(schedule: [Float], cfgScale: Float) {
        self.schedule = schedule
        self.cfgScale = cfgScale
    }

    init(numSteps: Int, shift: Double, cfgScale: Float) throws {
        self.schedule = try buildFlowSchedule(numSteps: numSteps, maxSteps: 100, shift: shift)
        self.cfgScale = cfgScale
    }

    var numSteps: Int { schedule.count }

    /// Run the full denoising loop.
    ///
    /// - Parameters:
    ///   - noise:                 [B, T, audioAcousticHiddenDim] initial Gaussian noise
    ///   - contextLatents:        [B, T, 128] context (silence/src latents + chunk masks)
    ///   - encoderHiddenStates:   [B, S, hiddenSize] conditional encoder output
    ///   - encoderAttentionMask:  [B, S] int 0/1 packed pad mask, or `nil`
    ///   - nullConditionEmb:      [1, 1, hiddenSize] unconditional embedding
    ///   - model:                 The DiT decoder
    ///   - onStep:                Progress callback (step index, total steps)
    /// - Returns: Denoised acoustic latent [B, T, audioAcousticHiddenDim]
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
        var xt = noise

        // Unconditional mask: single valid token per sample.
        let nullMask = MLXArray.ones([B, 1]).asType(.int32)
        // Broadcast null emb to batch size if needed.
        let nullEmb: MLXArray = B > 1
            ? tiled(nullConditionEmb, repetitions: [B, 1, 1])
            : nullConditionEmb

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
            let vUncond = model(
                hiddenStates:         xt,
                contextLatents:       contextLatents,
                timestep:             tTensor,
                timestepR:            tTensor,
                encoderHiddenStates:  nullEmb,
                encoderAttentionMask: nullMask
            )

            let vt = vUncond + MLXArray(cfgScale) * (vCond - vUncond)

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

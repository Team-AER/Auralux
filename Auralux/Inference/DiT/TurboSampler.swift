import MLX
import MLXNN
import Foundation

// MARK: - Turbo timestep schedule (shift=3, 8 NFE — from generate_audio in modeling_acestep_v15_turbo.py)

private let kTurboSchedule: [Float] = [
    1.0,
    0.9545454545454546,
    0.9,
    0.8333333333333334,
    0.75,
    0.6428571428571429,
    0.5,
    0.3,
]

// MARK: - TurboSampler

/// 8-step ODE (Euler) sampler for ACE-Step v1.5 Turbo.
///
/// Flow-matching prediction: the model outputs a velocity field v(x_t, t).
/// ODE update: x_{t+1} = x_t − v * dt
/// Final step:  x_0 = x_t − v * t   (i.e. get_x0_from_noise)
struct TurboSampler {
    let schedule: [Float]

    init(schedule: [Float] = kTurboSchedule) {
        self.schedule = schedule
    }

    var numSteps: Int { schedule.count }

    /// Run the full denoising loop.
    ///
    /// - Parameters:
    ///   - noise:                [B, T, audioAcousticHiddenDim] initial Gaussian noise
    ///   - contextLatents:       [B, T, 128] context (silence/src latents + chunk masks)
    ///   - encoderHiddenStates:  [B, S, hiddenSize] from lyric/condition encoder
    ///   - model:                The DiT decoder
    ///   - onStep:               Progress callback (step index, total steps)
    /// - Returns: Denoised acoustic latent [B, T, audioAcousticHiddenDim]
    func sample(
        noise: MLXArray,
        contextLatents: MLXArray,
        encoderHiddenStates: MLXArray,
        model: AceStepDiTModel,
        onStep: ((Int, Int) -> Void)? = nil
    ) -> MLXArray {
        let B  = noise.shape[0]
        var xt = noise

        for (i, t) in schedule.enumerated() {
            let tTensor = MLXArray(Array(repeating: t, count: B))

            let vt = model.callAsFunction(
                hiddenStates:        xt,
                contextLatents:      contextLatents,
                timestep:            tTensor,
                timestepR:           tTensor,
                encoderHiddenStates: encoderHiddenStates
            )

            if i == schedule.count - 1 {
                // get_x0_from_noise: x0 = xt - vt * t
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

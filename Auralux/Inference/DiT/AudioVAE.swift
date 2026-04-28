import MLX
import MLXNN
import Foundation

// MARK: - Protocol

protocol AudioVAEDecoder: AnyObject {
    /// Decode DiT acoustic latents to a stereo waveform.
    /// - latent: [B, T, 64] from TurboSampler at 25 Hz
    /// - Returns: [B, T × 1920, 2] float32 PCM at 48 kHz
    func decode(latent: MLXArray) -> MLXArray
}

// MARK: - Snake1d  (x + sin²(exp(α)·x) / exp(β))
// Learned per-channel activation used throughout the Oobleck decoder.

final class Snake1d: Module, @unchecked Sendable {
    var alpha: MLXArray   // [C]
    var beta: MLXArray    // [C]

    init(channels: Int) {
        alpha = MLXArray.zeros([channels])
        beta  = MLXArray.zeros([channels])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, L, C] — broadcast alpha/beta over batch and length
        let a = exp(alpha).reshaped([1, 1, -1])
        let b = exp(beta).reshaped([1, 1, -1])
        let s = sin(a * x)
        return x + (s * s) / (b + 1e-9)
    }
}

// MARK: - OobleckResUnit  (Snake → dilated Conv7 → Snake → Conv1 + residual)

final class OobleckResUnit: Module, @unchecked Sendable {
    let snake1: Snake1d
    let conv1:  Conv1d
    let snake2: Snake1d
    let conv2:  Conv1d

    init(channels: Int, dilation: Int) {
        snake1 = Snake1d(channels: channels)
        conv1  = Conv1d(
            inputChannels:  channels,
            outputChannels: channels,
            kernelSize:     7,
            padding:        3 * dilation,
            dilation:       dilation
        )
        snake2 = Snake1d(channels: channels)
        conv2  = Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(snake1(x))
        h = conv2(snake2(h))
        return x + h
    }
}

// MARK: - OobleckDecoderBlock  (Snake → ConvTransposed → 3 × ResUnit[dil 1,3,9])

final class OobleckDecoderBlock: Module, @unchecked Sendable {
    let snake1:   Snake1d
    let convT1:   ConvTransposed1d
    let resUnit1: OobleckResUnit
    let resUnit2: OobleckResUnit
    let resUnit3: OobleckResUnit

    init(inChannels: Int, outChannels: Int, stride: Int) {
        snake1   = Snake1d(channels: inChannels)
        convT1   = ConvTransposed1d(
            inputChannels:  inChannels,
            outputChannels: outChannels,
            kernelSize:     stride * 2,
            stride:         stride,
            padding:        stride / 2
        )
        resUnit1 = OobleckResUnit(channels: outChannels, dilation: 1)
        resUnit2 = OobleckResUnit(channels: outChannels, dilation: 3)
        resUnit3 = OobleckResUnit(channels: outChannels, dilation: 9)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convT1(snake1(x))
        h = resUnit1(h)
        h = resUnit2(h)
        h = resUnit3(h)
        return h
    }
}

// MARK: - DCHiFiGANDecoder  (Oobleck VAE decoder: latent → 48 kHz stereo audio)
//
// Architecture (AutoencoderOobleck from stable_audio_tools):
//   conv1(64→2048, k=7)
//   block[0]: 2048→1024, stride=10   (×10 upsample)
//   block[1]: 1024→512,  stride=6    (×6)
//   block[2]: 512→256,   stride=4    (×4)
//   block[3]: 256→128,   stride=4    (×4)
//   block[4]: 128→128,   stride=2    (×2)
//   snake1(128) → conv2(128→2, k=7, no bias)
//
// Total upsampling: 10×6×4×4×2 = 1920  →  25 Hz latents → 48 kHz audio

final class DCHiFiGANDecoder: Module, AudioVAEDecoder, @unchecked Sendable {

    let conv1:  Conv1d
    let blocks: [OobleckDecoderBlock]
    let snake1: Snake1d
    let conv2:  Conv1d

    override init() {
        conv1  = Conv1d(inputChannels: 64,  outputChannels: 2048, kernelSize: 7, padding: 3)
        blocks = [
            OobleckDecoderBlock(inChannels: 2048, outChannels: 1024, stride: 10),
            OobleckDecoderBlock(inChannels: 1024, outChannels:  512, stride:  6),
            OobleckDecoderBlock(inChannels:  512, outChannels:  256, stride:  4),
            OobleckDecoderBlock(inChannels:  256, outChannels:  128, stride:  4),
            OobleckDecoderBlock(inChannels:  128, outChannels:  128, stride:  2),
        ]
        snake1 = Snake1d(channels: 128)
        conv2  = Conv1d(
            inputChannels:  128,
            outputChannels: 2,
            kernelSize:     7,
            padding:        3,
            bias:           false
        )
        super.init()
    }

    /// - latent: [B, T, 64] acoustic latent at 25 Hz
    /// - Returns: [B, T × 1920, 2] stereo PCM at 48 kHz
    func decode(latent: MLXArray) -> MLXArray {
        var x = latent.asType(.float32)
        x = conv1(x)
        for block in blocks { x = block(x) }
        return conv2(snake1(x)) // [B, T×1920, 2] stereo
    }
}

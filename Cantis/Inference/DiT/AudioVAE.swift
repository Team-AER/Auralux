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
// Learned per-channel activation used throughout the Oobleck encoder + decoder.

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

// MARK: - OobleckEncoderBlock  (3 × ResUnit[dil 1,3,9] → Snake → strided Conv)

final class OobleckEncoderBlock: Module, @unchecked Sendable {
    let resUnit1: OobleckResUnit
    let resUnit2: OobleckResUnit
    let resUnit3: OobleckResUnit
    let snake1:   Snake1d
    let conv1:    Conv1d

    init(inChannels: Int, outChannels: Int, stride: Int) {
        resUnit1 = OobleckResUnit(channels: inChannels, dilation: 1)
        resUnit2 = OobleckResUnit(channels: inChannels, dilation: 3)
        resUnit3 = OobleckResUnit(channels: inChannels, dilation: 9)
        snake1   = Snake1d(channels: inChannels)
        // Mirrors PyTorch `padding=math.ceil(stride/2)` from
        // `diffusers/models/autoencoders/autoencoder_oobleck.py:OobleckEncoderBlock`.
        conv1    = Conv1d(
            inputChannels:  inChannels,
            outputChannels: outChannels,
            kernelSize:     stride * 2,
            stride:         stride,
            padding:        (stride + 1) / 2
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = resUnit1(x)
        h = resUnit2(h)
        h = resUnit3(h)
        h = snake1(h)
        return conv1(h)
    }
}

// MARK: - DCHiFiGANDecoder  (Oobleck VAE decoder: latent → 48 kHz stereo audio)
//
// Architecture (AutoencoderOobleck from stable_audio_tools):
//   conv1(64→2048, k=7)
//   blocks[0]: 2048→1024, stride=10   (×10 upsample)
//   blocks[1]: 1024→512,  stride=6    (×6)
//   blocks[2]: 512→256,   stride=4    (×4)
//   blocks[3]: 256→128,   stride=4    (×4)
//   blocks[4]: 128→128,   stride=2    (×2)
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

    // Latent frames per decode chunk (~2 s of audio) and border overlap.
    // Without chunking, the last Oobleck block materialises [B, T*1920, 128]
    // which exceeds 1 GB for clips longer than ~30 s.
    private static let kChunkFrames   = 50
    private static let kOverlapFrames = 16

    /// - latent: [B, T, 64] acoustic latent at 25 Hz
    /// - Returns: [B, T × 1920, 2] stereo PCM at 48 kHz
    func decode(latent: MLXArray) -> MLXArray {
        let T = latent.shape[1]
        guard T > DCHiFiGANDecoder.kChunkFrames else {
            return decodeSlice(latent)
        }
        var chunks: [MLXArray] = []
        var start = 0
        while start < T {
            let end      = min(start + DCHiFiGANDecoder.kChunkFrames, T)
            let padLeft  = min(start, DCHiFiGANDecoder.kOverlapFrames)
            let padRight = min(T - end, DCHiFiGANDecoder.kOverlapFrames)
            var audio    = decodeSlice(latent[0..., (start - padLeft)..<(end + padRight), 0...])
            let aLen     = audio.shape[1]
            let trimL    = padLeft  * 1920
            let trimR    = padRight * 1920
            audio = audio[0..., trimL..<(aLen - trimR), 0...]
            eval(audio)
            MLX.Memory.clearCache()
            chunks.append(audio)
            start = end
        }
        return concatenated(chunks, axis: 1)
    }

    private func decodeSlice(_ latent: MLXArray) -> MLXArray {
        var x = latent.asType(.float32)
        x = conv1(x)
        for block in blocks { x = block(x) }
        return conv2(snake1(x))
    }
}

// MARK: - DCHiFiGANEncoder  (Oobleck VAE encoder: 48 kHz stereo audio → latent)
//
// Architecture (mirror of decoder, downsampling instead of upsampling):
//   conv1(2→128, k=7)
//   blocks[0]: 128→128,  stride=2
//   blocks[1]: 128→256,  stride=4
//   blocks[2]: 256→512,  stride=4
//   blocks[3]: 512→1024, stride=6
//   blocks[4]: 1024→2048, stride=10
//   snake1(2048) → conv2(2048→128, k=3, p=1)
//
// Total downsampling: 2×4×4×6×10 = 1920  →  48 kHz audio → 25 Hz pre-Gaussian.
// The 128-channel output is `[mean (64) | scale (64)]`. Inference takes the
// mode (i.e. `mean` = first 64 channels); see `OobleckDiagonalGaussianDistribution`
// in upstream `diffusers/models/autoencoders/autoencoder_oobleck.py`.

final class DCHiFiGANEncoder: Module, @unchecked Sendable {

    let conv1:  Conv1d
    let blocks: [OobleckEncoderBlock]
    let snake1: Snake1d
    let conv2:  Conv1d

    override init() {
        conv1  = Conv1d(inputChannels: 2, outputChannels: 128, kernelSize: 7, padding: 3)
        blocks = [
            OobleckEncoderBlock(inChannels:  128, outChannels:  128, stride:  2),
            OobleckEncoderBlock(inChannels:  128, outChannels:  256, stride:  4),
            OobleckEncoderBlock(inChannels:  256, outChannels:  512, stride:  4),
            OobleckEncoderBlock(inChannels:  512, outChannels: 1024, stride:  6),
            OobleckEncoderBlock(inChannels: 1024, outChannels: 2048, stride: 10),
        ]
        snake1 = Snake1d(channels: 2048)
        conv2  = Conv1d(inputChannels: 2048, outputChannels: 128, kernelSize: 3, padding: 1)
        super.init()
    }

    private static let kChunkFrames   = 50
    private static let kOverlapFrames = 16

    /// Encode 48 kHz stereo audio into the 25 Hz acoustic latent.
    ///
    /// - audio: `[B, T_audio, 2]` float32 in [-1, 1]. Length must be a multiple
    ///   of 1920 — pad with zeros at the call site if needed.
    /// - Returns: `[B, T_audio / 1920, 64]` — the *mean* of the diagonal
    ///   Gaussian. Use this directly as the DiT acoustic latent.
    func encode(audio: MLXArray) -> MLXArray {
        let T = audio.shape[1] / 1920
        guard T > DCHiFiGANEncoder.kChunkFrames else {
            return encodeSlice(audio)
        }
        var chunks: [MLXArray] = []
        var start = 0
        while start < T {
            let end      = min(start + DCHiFiGANEncoder.kChunkFrames, T)
            let padLeft  = min(start, DCHiFiGANEncoder.kOverlapFrames)
            let padRight = min(T - end, DCHiFiGANEncoder.kOverlapFrames)
            var lat      = encodeSlice(audio[0..., (start - padLeft) * 1920..<(end + padRight) * 1920, 0...])
            let latLen   = lat.shape[1]
            lat = lat[0..., padLeft..<(latLen - padRight), 0...]
            eval(lat)
            MLX.Memory.clearCache()
            chunks.append(lat)
            start = end
        }
        return concatenated(chunks, axis: 1)
    }

    private func encodeSlice(_ audio: MLXArray) -> MLXArray {
        var x = audio.asType(.float32)
        x = conv1(x)
        for block in blocks { x = block(x) }
        x = conv2(snake1(x))
        return x[0..., 0..., 0..<64]
    }
}

// MARK: - DCHiFiGANVAE  (top-level wrapper exposing encode + decode)

/// Top-level Oobleck VAE wrapper. Holds both `encoder` and `decoder`
/// sub-modules so the converted weight keys (`encoder.*`, `decoder.*`)
/// load directly with no extra remapping.
final class DCHiFiGANVAE: Module, AudioVAEDecoder, @unchecked Sendable {
    let encoder: DCHiFiGANEncoder
    let decoder: DCHiFiGANDecoder

    override init() {
        encoder = DCHiFiGANEncoder()
        decoder = DCHiFiGANDecoder()
        super.init()
    }

    /// - latent: [B, T, 64] acoustic latent at 25 Hz
    /// - Returns: [B, T × 1920, 2] stereo PCM at 48 kHz
    func decode(latent: MLXArray) -> MLXArray {
        decoder.decode(latent: latent)
    }

    /// - audio:  [B, T_audio, 2] float32 stereo PCM at 48 kHz, length multiple of 1920
    /// - Returns: [B, T_audio / 1920, 64] acoustic latent at 25 Hz
    func encode(audio: MLXArray) -> MLXArray {
        encoder.encode(audio: audio)
    }
}

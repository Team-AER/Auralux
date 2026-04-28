import Foundation
import MLX

/// Loads ACE-Step's text-to-music reference latent.
///
/// Upstream stores `silence_latent.pt` as `[1, 64, T]`; the MLX artifact is
/// normalized to `[1, T, 64]` so it can be concatenated with chunk masks.
enum SilenceLatentLoader {

    static func load(baseDir: URL) throws -> MLXArray {
        let url = baseDir
            .appendingPathComponent("dit")
            .appendingPathComponent("silence_latent.safetensors")

        guard FileManager.default.fileExists(atPath: url.path) else {
            throw NativeEngineError.weightsNotFound(url)
        }

        let arrays = try loadArrays(url: url)
        guard let latent = arrays["silence_latent"] ?? arrays["silenceLatent"] else {
            throw NativeEngineError.generationFailed("silence_latent.safetensors does not contain silence_latent")
        }

        return normalize(latent)
    }

    static func slice(_ latent: MLXArray, frames: Int) throws -> MLXArray {
        guard frames > 0 else {
            throw NativeEngineError.generationFailed("Requested silence latent with non-positive frame count")
        }
        guard latent.shape.count == 3, latent.shape[0] == 1, latent.shape[2] == 64 else {
            throw NativeEngineError.generationFailed("Silence latent must have shape [1, T, 64], got \(latent.shape)")
        }

        let available = latent.shape[1]
        guard available > 0 else {
            throw NativeEngineError.generationFailed("Silence latent is empty")
        }

        if frames <= available {
            return latent[0..., ..<frames, 0...]
        }

        let repeats = (frames + available - 1) / available
        let tiled = concatenated(Array(repeating: latent, count: repeats), axis: 1)
        return tiled[0..., ..<frames, 0...]
    }

    private static func normalize(_ latent: MLXArray) -> MLXArray {
        switch latent.shape {
        case let shape where shape.count == 3 && shape[0] == 1 && shape[2] == 64:
            return latent.asType(.float32)
        case let shape where shape.count == 3 && shape[0] == 1 && shape[1] == 64:
            return latent.transposed(0, 2, 1).asType(.float32)
        case let shape where shape.count == 2 && shape[1] == 64:
            return latent.reshaped([1, shape[0], 64]).asType(.float32)
        case let shape where shape.count == 2 && shape[0] == 64:
            return latent.transposed(1, 0).reshaped([1, shape[1], 64]).asType(.float32)
        default:
            return latent.asType(.float32)
        }
    }
}

import MLX
import MLXNN
import Foundation

/// Loads converted VAE encoder + decoder weights from `<baseDir>/vae/vae_weights.safetensors`.
enum VAEWeightLoader {

    static func load(baseDir: URL, into model: DCHiFiGANVAE) throws {
        let url = baseDir
            .appendingPathComponent("vae")
            .appendingPathComponent("vae_weights.safetensors")

        guard FileManager.default.fileExists(atPath: url.path) else {
            throw NativeEngineError.weightsNotFound(url)
        }

        let flat   = try loadArrays(url: url)
        let nested = ModuleParameters.unflattened(flat)
        #if DEBUG
        try model.update(parameters: nested, verify: .shapeMismatch)
        #else
        try model.update(parameters: nested, verify: .none)
        #endif
        eval(model.parameters())
    }
}

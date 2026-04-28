import MLX
import MLXNN
import Foundation

/// Loads converted LM weights from a safetensors file produced by tools/convert_weights.py.
///
/// Expected file: `checkpoints/ace-step-v1.5-mlx/lm/lm_weights.safetensors`
/// All keys in the file are already camelCase Swift module paths
/// (e.g. `layers.0.selfAttn.qProj.weight`) — no remapping needed here.
enum LMWeightLoader {

    static func load(from url: URL, into model: ACEStepLMModel) throws {
        let flat    = try loadArrays(url: url)
        let nested  = ModuleParameters.unflattened(flat)
        #if DEBUG
        try model.update(parameters: nested, verify: .shapeMismatch)
        #else
        try model.update(parameters: nested, verify: .none)
        #endif
        eval(model.parameters())
    }

    /// Convenience: load from the default output directory relative to `baseDir`.
    static func load(baseDir: URL, into model: ACEStepLMModel) throws {
        let url = baseDir
            .appendingPathComponent("lm")
            .appendingPathComponent("lm_weights.safetensors")
        try load(from: url, into: model)
    }
}

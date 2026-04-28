import MLX
import MLXNN
import Foundation

/// Loads converted Qwen3-Embedding-0.6B weights from
/// `<baseDir>/text/text_weights.safetensors` into a `Qwen3EncoderModel`.
///
/// Conversion (`tools/convert_weights.py`) drops the `model.` prefix and
/// preserves the upstream key layout under `_remap_text_encoder_key`.
enum Qwen3EncoderWeightLoader {

    static func load(from url: URL, into model: Qwen3EncoderModel) throws {
        let flat   = try loadArrays(url: url)
        let nested = ModuleParameters.unflattened(flat)
        #if DEBUG
        try model.update(parameters: nested, verify: .shapeMismatch)
        #else
        try model.update(parameters: nested, verify: .none)
        #endif
        eval(model.parameters())
    }

    static func load(baseDir: URL, into model: Qwen3EncoderModel) throws {
        let url = baseDir
            .appendingPathComponent("text")
            .appendingPathComponent("text_weights.safetensors")
        try load(from: url, into: model)
    }
}

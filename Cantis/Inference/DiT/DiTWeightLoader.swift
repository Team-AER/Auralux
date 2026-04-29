import MLX
import MLXNN
import Foundation

/// Loads converted DiT weights from safetensors produced by tools/convert_weights.py.
///
/// Expected file: <baseDir>/dit/dit_weights.safetensors
///
/// The converter remaps checkpoint keys to match the Swift module hierarchy:
///   decoder.*                     → decoder.*
///   encoder.lyric_encoder.*       → lyricEncoder.*
///   detokenizer.*                 → detokenizer.*
///   null_condition_emb            → nullConditionEmb
///   tokenizer.*                   → (skipped)
enum DiTWeightLoader {

    static func load(from url: URL, into model: ACEStepDiT) throws {
        let flat = try loadArrays(url: url)
        let nested = ModuleParameters.unflattened(flat)
        try model.update(parameters: nested, verify: .none)
        eval(model.parameters())
    }

    static func load(baseDir: URL, into model: ACEStepDiT) throws {
        let url = baseDir
            .appendingPathComponent("dit")
            .appendingPathComponent("dit_weights.safetensors")
        try load(from: url, into: model)
    }
}

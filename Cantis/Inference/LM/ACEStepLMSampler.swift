import MLX
import Foundation

/// Driver for the 5 Hz audio-code language model.
///
/// **Status: not yet wired end-to-end.**
///
/// The v1.5 turbo `acestep-5Hz-lm-0.6B` is a Qwen2-style decoder LM whose
/// vocabulary serialises FSQ audio codes as discrete text-style tokens. The
/// authoritative encoding format (special-token boundaries between caption /
/// lyrics / audio, end-of-stream markers, code-row interleaving) lives in
/// upstream training/inference scripts that aren't in the repo we shipped
/// alongside `modeling_acestep_v15_turbo.py`. Implementing it without that
/// reference would be guesswork — see the saved feedback memory
/// `feedback_verify_against_upstream`.
///
/// Until those scripts are available, calling this throws a clear error so
/// the UI can surface it. The tensor surface is in place: once
/// `ResidualFSQ.getOutputFromIndices(...)` receives an `[1, codeFrames, Q]`
/// integer tensor, the rest of the cover/text2musicLM pipeline follows.
enum ACEStepLMSampler {

    static func generate(
        lm: ACEStepLMModel,
        prompt: String,
        lyrics: String,
        language: String,
        codeFrames: Int,
        seed: Int?
    ) throws -> MLXArray {
        throw NativeEngineError.generationFailed(
            "text2musicLM mode is not implemented yet — the 5 Hz LM's audio-code "
          + "tokenisation format requires upstream scripts that aren't checked in."
        )
    }
}

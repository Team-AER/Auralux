import Foundation

/// What the DiT pipeline is being asked to do for a given generation.
///
/// Mirrors the columns in the ACE-Step v1.5 feature matrix
/// (Text2Music / Cover / Repaint / Extract). `text2musicLM` is our local
/// addition for the LM-driven hint path that the upstream calls "audio_codes".
///
/// `text2music`, `cover`, `repaint`, `extract` are wired end-to-end.
/// `text2musicLM` is opt-in once the 5 Hz LM toggle is on AND the
/// upstream-canonical audio-code tokeniser is implemented (see
/// `ACEStepLMSampler` for status).
enum GenerationMode: String, Codable, CaseIterable, Identifiable, Sendable {
    case text2music
    case text2musicLM
    case cover
    case repaint
    case `extract`

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .text2music:   return "Text → Music"
        case .text2musicLM: return "Text → Music (LM hints)"
        case .cover:        return "Cover"
        case .repaint:      return "Repaint"
        case .extract:      return "Extract"
        }
    }

    /// Whether the engine currently knows how to run this mode.
    var isImplemented: Bool {
        switch self {
        case .text2music, .extract, .cover, .repaint: return true
        case .text2musicLM:                            return false
        }
    }

    var requiresSourceAudio: Bool {
        self == .cover || self == .repaint
    }

    var requiresReferAudio: Bool {
        self == .extract || self == .cover
    }

    var requiresRepaintMask: Bool {
        self == .repaint
    }
}

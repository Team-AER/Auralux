import Foundation

/// Which ACE-Step DiT checkpoint the engine targets.
enum DiTVariant: String, Codable, CaseIterable, Identifiable, Sendable {
    case turbo
    case sft
    case base
    case xlTurbo = "xl-turbo"
    case xlSft   = "xl-sft"
    case xlBase  = "xl-base"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .turbo:   return "Turbo (8 steps, CFG-distilled)"
        case .sft:     return "SFT (60 steps)"
        case .base:    return "Base (60 steps)"
        case .xlTurbo: return "XL Turbo (8 steps, CFG-distilled)"
        case .xlSft:   return "XL SFT (60 steps)"
        case .xlBase:  return "XL Base (60 steps)"
        }
    }

    /// Local model directory name and HF repo suffix.
    var mlxDirectoryName: String {
        switch self {
        case .turbo:   return "ace-step-v1.5-mlx"
        case .sft:     return "ace-step-v1.5-sft-mlx"
        case .base:    return "ace-step-v1.5-base-mlx"
        case .xlTurbo: return "ace-step-v1.5-xl-turbo-mlx"
        case .xlSft:   return "ace-step-v1.5-xl-sft-mlx"
        case .xlBase:  return "ace-step-v1.5-xl-base-mlx"
        }
    }

    /// Turbo variants bake CFG into the distillation and ignore `cfgScale > 1`.
    var usesCFGDistillation: Bool {
        switch self {
        case .turbo, .xlTurbo: return true
        default: return false
        }
    }

    /// Whether CFG scale is meaningful for this variant.
    var respectsCFG: Bool { !usesCFGDistillation }

    /// AceStepConfig preset for this variant.
    var modelConfig: AceStepConfig {
        switch self {
        case .turbo:   return .turbo
        case .sft:     return .sft
        case .base:    return .base
        case .xlTurbo: return .xlTurbo
        case .xlSft:   return .xlSft
        case .xlBase:  return .xlBase
        }
    }

    /// Hard upper bound enforced by the sampler for this variant.
    var maxNumSteps: Int {
        usesCFGDistillation ? 20 : 100
    }

    /// Upstream Gradio UI defaults: turbo=8, base/SFT=60.
    var defaultNumSteps: Int {
        usesCFGDistillation ? 8 : 60
    }

    /// Upstream pipeline defaults: turbo is CFG-distilled (1.0), base/SFT use 15.0.
    var defaultCfgScale: Double {
        usesCFGDistillation ? 1.0 : 15.0
    }

    /// Whether converted MLX weights exist on HF and the engine can load this variant.
    var isAvailable: Bool {
        switch self {
        case .turbo, .sft, .base: return true
        default: return false
        }
    }

    /// Whether the app can download this variant directly (vs. requiring the conversion script).
    var canDownloadInApp: Bool {
        switch self {
        case .turbo, .sft, .base: return true
        default: return false
        }
    }

    /// Non-turbo variants symlink lm/, vae/, text/ from the turbo directory.
    var requiresTurboBase: Bool { self != .turbo }
}

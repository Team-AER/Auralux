import Foundation

/// A user-added ACE-Step-compatible model. Inherits architecture/config from a
/// built-in `baseVariant` (turbo / sft / base) — the source repo or folder must
/// follow the ACE-Step MLX file layout (dit/, lm/, vae/, text/).
struct CustomModel: Identifiable, Codable, Hashable, Sendable {
    enum Source: Codable, Hashable, Sendable {
        case huggingFace(repoID: String)
        case localFolder(absolutePath: String)
    }

    let id: String
    var displayName: String
    var source: Source
    var baseVariant: DiTVariant
    var addedAt: Date

    /// Where the model's safetensors live on disk. HF-managed models live under
    /// `Models/custom-<id>/`; local-folder models point at the user-chosen path.
    var localDirectory: URL {
        switch source {
        case .huggingFace:
            return FileUtilities.modelDirectory.appendingPathComponent("custom-\(id)", isDirectory: true)
        case .localFolder(let path):
            return URL(fileURLWithPath: path, isDirectory: true)
        }
    }

    var isHFManaged: Bool {
        if case .huggingFace = source { return true }
        return false
    }

    var sourceDescription: String {
        switch source {
        case .huggingFace(let repo): return repo
        case .localFolder(let path): return (path as NSString).abbreviatingWithTildeInPath
        }
    }
}

/// Identifier used by the engine to refer to either a built-in variant or a
/// custom model — needed to track which one is currently downloading.
enum ModelID: Equatable, Hashable, Sendable {
    case builtin(DiTVariant)
    case custom(String)
}

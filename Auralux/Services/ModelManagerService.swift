import Foundation

/// A single downloadable model variant hosted on HuggingFace.
struct ModelArtifact: Identifiable, Codable, Hashable, Sendable {
    var id: String { name }
    var name: String
    var repoID: String
    var description: String
    var estimatedSizeGB: Double
}

/// Registry of ACE-Step model variants available under Team-AER.
///
/// `mlxArtifact` is the one the app actually downloads; the others are listed
/// for reference in Settings and can be supported once their weights are converted.
enum ModelManagerService {

    // MARK: - Active MLX repo

    /// The converted MLX model the app downloads via ModelDownloader.
    static let mlxRepoID = "Team-AER/ace-step-v1.5-mlx"

    /// Contents of the MLX repo (matches ModelDownloader.manifest).
    static let mlxArtifact = ModelArtifact(
        name: "ace-step-v1.5-turbo-mlx",
        repoID: mlxRepoID,
        description: "ACE-Step v1.5 Turbo — 8-step flow-matching, MLX-native (DiT 2B + LM 0.6B)",
        estimatedSizeGB: 5.2
    )

    // MARK: - Upstream PyTorch variants (reference only — not downloaded in-app)

    static let upstreamVariants: [ModelArtifact] = [
        ModelArtifact(
            name: "acestep-v15-turbo",
            repoID: "ACE-Step/Ace-Step1.5",
            description: "DiT 2B Turbo — 8-step inference (upstream PyTorch checkpoint)",
            estimatedSizeGB: 4.8
        ),
        ModelArtifact(
            name: "acestep-v15-sft",
            repoID: "ACE-Step/Ace-Step1.5",
            description: "DiT 2B SFT — 50-step, higher quality than turbo (upstream PyTorch)",
            estimatedSizeGB: 4.8
        ),
        ModelArtifact(
            name: "acestep-5Hz-lm-0.6B",
            repoID: "ACE-Step/acestep-5Hz-lm-0.6B",
            description: "LM 0.6B — CoT metadata + prompt rewriting (upstream PyTorch)",
            estimatedSizeGB: 1.2
        ),
    ]

    static let knownArtifacts: [ModelArtifact] = [mlxArtifact] + upstreamVariants
}

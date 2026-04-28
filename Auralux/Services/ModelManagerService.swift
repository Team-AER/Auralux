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

    // MARK: - Additional MLX repos (DiT-only; share VAE/LM/text with turbo via symlinks)

    static let sftMLXArtifact = ModelArtifact(
        name: "ace-step-v1.5-sft-mlx",
        repoID: "Team-AER/ace-step-v1.5-sft-mlx",
        description: "ACE-Step v1.5 SFT — 60-step flow-matching, MLX-native (DiT 2B; shares VAE/LM/text with turbo)",
        estimatedSizeGB: 3.5
    )
    static let baseMLXArtifact = ModelArtifact(
        name: "ace-step-v1.5-base-mlx",
        repoID: "Team-AER/ace-step-v1.5-base-mlx",
        description: "ACE-Step v1.5 Base — 60-step flow-matching, MLX-native (DiT 2B; shares VAE/LM/text with turbo)",
        estimatedSizeGB: 3.5
    )
    static let xlTurboMLXArtifact = ModelArtifact(
        name: "ace-step-v1.5-xl-turbo-mlx",
        repoID: "Team-AER/ace-step-v1.5-xl-turbo-mlx",
        description: "ACE-Step v1.5 XL Turbo — 8-step CFG-distilled, MLX-native (DiT 5B + LM 0.6B)",
        estimatedSizeGB: 11.0
    )
    static let xlSftMLXArtifact = ModelArtifact(
        name: "ace-step-v1.5-xl-sft-mlx",
        repoID: "Team-AER/ace-step-v1.5-xl-sft-mlx",
        description: "ACE-Step v1.5 XL SFT — 60-step flow-matching, MLX-native (DiT 5B; shares VAE/LM/text with turbo)",
        estimatedSizeGB: 9.0
    )
    static let xlBaseMLXArtifact = ModelArtifact(
        name: "ace-step-v1.5-xl-base-mlx",
        repoID: "Team-AER/ace-step-v1.5-xl-base-mlx",
        description: "ACE-Step v1.5 XL Base — 60-step flow-matching, MLX-native (DiT 5B; shares VAE/LM/text with turbo)",
        estimatedSizeGB: 9.0
    )

    /// All MLX-native artifacts that the app can load, in display order.
    static let mlxArtifacts: [ModelArtifact] = [
        mlxArtifact, sftMLXArtifact, baseMLXArtifact,
        xlTurboMLXArtifact, xlSftMLXArtifact, xlBaseMLXArtifact
    ]

    static let knownArtifacts: [ModelArtifact] = mlxArtifacts
}

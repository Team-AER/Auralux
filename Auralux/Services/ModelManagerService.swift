import Foundation

/// Represents an ACE-Step model component that can be managed.
struct ModelArtifact: Identifiable, Codable, Hashable, Sendable {
    var id: String { name }
    var name: String
    var repoID: String
    var description: String
    var estimatedSizeGB: Double
}

/// Represents the overall model readiness state, derived from the server health.
struct ModelStatus: Sendable {
    var ditLoaded: Bool
    var llmLoaded: Bool
    var device: String
    var engine: String
    var ditModel: String
    var llmModel: String
    var error: String?

    static let unknown = ModelStatus(
        ditLoaded: false,
        llmLoaded: false,
        device: "unknown",
        engine: "unknown",
        ditModel: "",
        llmModel: "",
        error: nil
    )
}

enum ModelManagerError: Error, LocalizedError {
    case serverNotRunning
    case downloadFailed(String)

    var errorDescription: String? {
        switch self {
        case .serverNotRunning:
            return "Inference server is not running. Start it first."
        case .downloadFailed(let detail):
            return "Model download failed: \(detail)"
        }
    }
}

/// Manages model lifecycle by communicating with the local inference server.
///
/// ACE-Step 1.5 auto-downloads models from HuggingFace on first use.
/// This service provides a Swift-friendly interface to check readiness
/// and trigger downloads via the server's REST endpoints.
actor ModelManagerService {
    private let inferenceService: InferenceService

    init(inferenceService: InferenceService = InferenceService()) {
        self.inferenceService = inferenceService
    }

    /// The directory where ACE-Step stores model checkpoints.
    var checkpointDirectory: URL {
        FileUtilities.modelDirectory
    }

    /// Checks server health to determine model status.
    func fetchModelStatus() async -> ModelStatus {
        guard let health = await inferenceService.fetchHealth() else {
            return .unknown
        }
        return ModelStatus(
            ditLoaded: health.modelLoaded,
            llmLoaded: health.llmLoaded ?? false,
            device: health.device ?? "unknown",
            engine: health.engine ?? "unknown",
            ditModel: health.ditModel ?? "",
            llmModel: health.llmModel ?? "",
            error: health.modelError
        )
    }

    func isServerHealthy() async -> Bool {
        await inferenceService.isHealthy()
    }

    /// Whether the server is running and has the DiT model loaded.
    func isModelReady() async -> Bool {
        let status = await fetchModelStatus()
        return status.ditLoaded
    }

    /// Triggers model download via the server.
    /// ACE-Step 1.5 downloads from HuggingFace automatically.
    func triggerModelDownload() async throws {
        try await inferenceService.triggerModelDownload()
    }

    /// Returns the list of known model artifacts for display in settings.
    static let knownArtifacts: [ModelArtifact] = [
        ModelArtifact(
            name: "acestep-v15-turbo",
            repoID: "ACE-Step/Ace-Step1.5",
            description: "DiT turbo model — 8-step inference, commercial-grade quality",
            estimatedSizeGB: 2.5
        ),
        ModelArtifact(
            name: "acestep-5Hz-lm-0.6B",
            repoID: "ACE-Step/acestep-5Hz-lm-0.6B",
            description: "5Hz language model (0.6B) — metadata generation and CoT reasoning",
            estimatedSizeGB: 1.2
        ),
        ModelArtifact(
            name: "vae",
            repoID: "ACE-Step/Ace-Step1.5",
            description: "VAE decoder — converts latents to audio waveforms",
            estimatedSizeGB: 0.3
        ),
    ]
}

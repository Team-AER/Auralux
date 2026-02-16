import Foundation

// MARK: - Request / Response types

struct GenerationRequest: Codable, Sendable {
    var prompt: String
    var lyrics: String
    var tags: [String]
    var duration: TimeInterval
    var variance: Double
    var seed: Int?
}

struct GenerationResponse: Codable, Sendable {
    var jobID: String
    var status: String
    var message: String?
}

struct GenerationStatusResponse: Codable, Sendable {
    var jobID: String
    var status: String
    var progress: Double
    var message: String?
    var audioPath: String?
}

struct HealthResponse: Codable, Sendable {
    var status: String
    var modelLoaded: Bool
    var llmLoaded: Bool?
    var modelError: String?
    var device: String?
    var engine: String?
    var ditModel: String?
    var llmModel: String?
}

// MARK: - Errors

enum InferenceError: Error, LocalizedError {
    case serverScriptMissing
    case serverLaunchFailed
    case serverUnhealthy
    case invalidResponse
    case requestFailed(String)
    case modelNotReady
    case sandboxRestricted

    var errorDescription: String? {
        switch self {
        case .serverScriptMissing:
            return "Server launch script not found. Run setup_env.sh first."
        case .serverLaunchFailed:
            return "Failed to start the inference server process."
        case .serverUnhealthy:
            return "Inference server did not become healthy in time."
        case .invalidResponse:
            return "Received an invalid response from the server."
        case .requestFailed(let detail):
            return "Request failed: \(detail)"
        case .modelNotReady:
            return "Model is still downloading or not yet loaded."
        case .sandboxRestricted:
            return "Cannot launch server in App Sandbox. Start the Auralux Engine manually."
        }
    }
}

// MARK: - Server Launcher Protocol

/// Abstracts how the inference server is started. Allows swapping between
/// Process-based (dev/direct distribution) and XPC-based (App Store) strategies.
protocol ServerLauncher: Sendable {
    func launch() async throws
    func stop() async
    var isRunning: Bool { get async }
}

/// Launches the Python inference server as a subprocess.
/// Works for development and direct (notarized) distribution.
/// NOT compatible with the Mac App Store sandbox.
final class ProcessServerLauncher: ServerLauncher, @unchecked Sendable {
    private var process: Process?
    private let lock = NSLock()

    var isRunning: Bool {
        lock.withLock { process?.isRunning ?? false }
    }

    func launch() async throws {
        if isRunning { return }

        guard !Self.isAppSandboxed else {
            throw InferenceError.sandboxRestricted
        }

        guard let scriptURL = Self.locateServerScript(),
              FileManager.default.fileExists(atPath: scriptURL.path) else {
            throw InferenceError.serverScriptMissing
        }

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/bin/zsh")
        proc.arguments = [scriptURL.path]
        proc.currentDirectoryURL = scriptURL.deletingLastPathComponent()
        proc.standardOutput = Pipe()
        proc.standardError = Pipe()

        do {
            try proc.run()
            lock.withLock { process = proc }
        } catch {
            throw InferenceError.serverLaunchFailed
        }
    }

    func stop() async {
        lock.withLock {
            guard let proc = process else { return }
            if proc.isRunning { proc.terminate() }
            process = nil
        }
    }

    /// Detects whether the current process is running inside the App Sandbox.
    private static var isAppSandboxed: Bool {
        ProcessInfo.processInfo.environment["APP_SANDBOX_CONTAINER_ID"] != nil
    }

    private static func locateServerScript() -> URL? {
        let candidates = [
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent("AuraluxEngine/start_api_server_macos.sh"),
            Bundle.main.bundleURL
                .deletingLastPathComponent()
                .appendingPathComponent("AuraluxEngine/start_api_server_macos.sh"),
            Bundle.main.resourceURL?
                .appendingPathComponent("AuraluxEngine/start_api_server_macos.sh"),
        ].compactMap { $0 }

        return candidates.first { FileManager.default.fileExists(atPath: $0.path) }
    }
}

/// Placeholder for a future XPC-based launcher suitable for the Mac App Store.
/// The XPC service would be a separate target bundled inside the .app that
/// manages the Python runtime and inference server.
final class XPCServerLauncher: ServerLauncher {
    var isRunning: Bool { false }

    func launch() async throws {
        throw InferenceError.sandboxRestricted
    }

    func stop() async {}
}

// MARK: - Service

actor InferenceService {
    private let baseURL: URL
    private let launcher: ServerLauncher

    init(
        baseURL: URL = AppConstants.inferenceBaseURL,
        launcher: ServerLauncher = ProcessServerLauncher()
    ) {
        self.baseURL = baseURL
        self.launcher = launcher
    }

    private func logInfo(_ msg: String) {
        Task { @MainActor in AppLogger.shared.info(msg, category: .inference) }
    }

    private func logError(_ msg: String) {
        Task { @MainActor in AppLogger.shared.error(msg, category: .inference) }
    }

    private func logDebug(_ msg: String) {
        Task { @MainActor in AppLogger.shared.debug(msg, category: .inference) }
    }

    // MARK: Server lifecycle

    func startServerIfNeeded() async throws {
        if await isHealthy() { return }
        if await launcher.isRunning { return }

        try await launcher.launch()

        let maxAttempts = 600
        for _ in 0..<maxAttempts {
            if await isHealthy() {
                return
            }
            try? await Task.sleep(for: .milliseconds(500))
        }
        throw InferenceError.serverUnhealthy
    }

    func stopServer() async {
        await launcher.stop()
    }

    // MARK: Health

    func isHealthy() async -> Bool {
        var request = URLRequest(url: baseURL.appendingPathComponent("health"))
        request.timeoutInterval = 5

        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse else { return false }
            return (200...299).contains(http.statusCode)
        } catch {
            return false
        }
    }

    func fetchHealth() async -> HealthResponse? {
        var request = URLRequest(url: baseURL.appendingPathComponent("health"))
        request.timeoutInterval = 5

        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse,
                  (200...299).contains(http.statusCode) else { return nil }
            return try JSONDecoder().decode(HealthResponse.self, from: data)
        } catch {
            return nil
        }
    }

    // MARK: Generation

    func generate(_ requestBody: GenerationRequest) async throws -> GenerationResponse {
        try await startServerIfNeeded()

        logInfo("POST /generate — prompt=\"\(requestBody.prompt.prefix(60))\"")

        var request = URLRequest(url: baseURL.appendingPathComponent("generate"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 30
        request.httpBody = try JSONEncoder().encode(requestBody)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw InferenceError.invalidResponse }
        guard (200...299).contains(http.statusCode) else {
            logError("generate failed: HTTP \(http.statusCode)")
            throw InferenceError.requestFailed("generate failed: \(http.statusCode)")
        }

        let decoded = try JSONDecoder().decode(GenerationResponse.self, from: data)
        logInfo("Job accepted: \(decoded.jobID)")
        return decoded
    }

    func poll(jobID: String) async throws -> GenerationStatusResponse {
        var request = URLRequest(url: baseURL.appendingPathComponent("jobs/\(jobID)"))
        request.timeoutInterval = 10
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw InferenceError.invalidResponse }
        guard (200...299).contains(http.statusCode) else {
            throw InferenceError.requestFailed("poll failed: \(http.statusCode)")
        }
        return try JSONDecoder().decode(GenerationStatusResponse.self, from: data)
    }

    func cancel(jobID: String) async throws {
        var request = URLRequest(url: baseURL.appendingPathComponent("jobs/\(jobID)/cancel"))
        request.httpMethod = "POST"
        let (_, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw InferenceError.invalidResponse }
        guard (200...299).contains(http.statusCode) else {
            throw InferenceError.requestFailed("cancel failed: \(http.statusCode)")
        }
    }

    // MARK: Model management

    func triggerModelDownload() async throws {
        try await startServerIfNeeded()

        var request = URLRequest(url: baseURL.appendingPathComponent("models/download"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let (_, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw InferenceError.invalidResponse }
        guard (200...299).contains(http.statusCode) else {
            throw InferenceError.requestFailed("model download trigger failed: \(http.statusCode)")
        }
    }
}

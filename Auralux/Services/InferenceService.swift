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
    var stats: EngineRuntimeStats?
}

struct EngineRuntimeStats: Codable, Equatable, Sendable {
    var pid: Int?
    var uptimeSeconds: Double?
    var cpuPercent: Double?
    var memoryRSSMB: Double?
    var activeThreads: Int?
    var jobCounts: [String: Int]?
    var statsError: String?

    static let empty = EngineRuntimeStats(
        pid: nil,
        uptimeSeconds: nil,
        cpuPercent: nil,
        memoryRSSMB: nil,
        activeThreads: nil,
        jobCounts: nil,
        statsError: nil
    )
}

// MARK: - Errors

enum InferenceError: Error, LocalizedError {
    case serverUnhealthy
    case invalidResponse
    case requestFailed(String)
    case jobNotFound(String)

    var errorDescription: String? {
        switch self {
        case .serverUnhealthy:
            return "Inference server did not become healthy in time."
        case .invalidResponse:
            return "Received an invalid response from the server."
        case .requestFailed(let detail):
            return "Request failed: \(detail)"
        case .jobNotFound(let jobID):
            return "Job \(jobID) lost — the inference server crashed and restarted. Please try again."
        }
    }
}

// MARK: - Service

actor InferenceService {
    private let baseURL: URL
    private let session: URLSession

    init(
        baseURL: URL = AppConstants.inferenceBaseURL,
        session: URLSession = .shared
    ) {
        self.baseURL = baseURL
        self.session = session
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

    // MARK: Server readiness

    func requireHealthyServer() async throws {
        if await isHealthy() { return }
        throw InferenceError.serverUnhealthy
    }

    // MARK: Health

    func isHealthy() async -> Bool {
        var request = URLRequest(url: baseURL.appendingPathComponent("health"))
        request.timeoutInterval = 5

        do {
            let (_, response) = try await session.data(for: request)
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
            let (data, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse,
                  (200...299).contains(http.statusCode) else { return nil }
            return try JSONDecoder().decode(HealthResponse.self, from: data)
        } catch {
            return nil
        }
    }

    // MARK: Generation

    func generate(_ requestBody: GenerationRequest) async throws -> GenerationResponse {
        try await requireHealthyServer()

        logInfo("POST /generate — prompt=\"\(requestBody.prompt.prefix(60))\"")

        var request = URLRequest(url: baseURL.appendingPathComponent("generate"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 30
        request.httpBody = try JSONEncoder().encode(requestBody)

        let (data, response) = try await session.data(for: request)
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
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw InferenceError.invalidResponse }
        if http.statusCode == 404 {
            throw InferenceError.jobNotFound(jobID)
        }
        guard (200...299).contains(http.statusCode) else {
            throw InferenceError.requestFailed("poll failed: \(http.statusCode)")
        }
        return try JSONDecoder().decode(GenerationStatusResponse.self, from: data)
    }

    func cancel(jobID: String) async throws {
        var request = URLRequest(url: baseURL.appendingPathComponent("jobs/\(jobID)/cancel"))
        request.httpMethod = "POST"
        let (_, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw InferenceError.invalidResponse }
        guard (200...299).contains(http.statusCode) else {
            throw InferenceError.requestFailed("cancel failed: \(http.statusCode)")
        }
    }

    // MARK: Model management

    func triggerModelDownload() async throws {
        try await requireHealthyServer()

        var request = URLRequest(url: baseURL.appendingPathComponent("models/download"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let (_, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw InferenceError.invalidResponse }
        guard (200...299).contains(http.statusCode) else {
            throw InferenceError.requestFailed("model download trigger failed: \(http.statusCode)")
        }
    }
}

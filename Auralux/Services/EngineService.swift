import Darwin
import Foundation
import Observation

/// Represents the overall state of the Auralux inference engine.
enum EngineState: Equatable, Sendable {
    case unknown
    case stopped
    case notSetup
    case settingUp(progress: String)
    case starting
    case running
    case ready
    case error(String)

    var isReady: Bool { self == .ready }
    var isRunning: Bool { self == .running || self == .ready }
    var isBusy: Bool {
        switch self {
        case .settingUp, .starting: return true
        default: return false
        }
    }
    var needsSetup: Bool { self == .notSetup }
    var isStopped: Bool { self == .stopped }
}

enum EnginePreparationError: LocalizedError {
    case setupRequired
    case busy
    case startupFailed(String)

    var errorDescription: String? {
        switch self {
        case .setupRequired:
            return "Set up the inference engine before generating audio."
        case .busy:
            return "The inference engine is busy. Try again in a moment."
        case .startupFailed(let message):
            return message
        }
    }
}

/// Manages the full lifecycle of the Auralux inference engine:
/// setup detection, environment provisioning, server start/stop,
/// health monitoring, and graceful shutdown.
@MainActor
@Observable
final class EngineService {
    var state: EngineState = .unknown
    var setupLog: [String] = []
    var modelStatus: ModelStatus = .unknown
    var runtimeStats: EngineRuntimeStats = .empty
    var lastHealthCheckAt: Date?
    var lastHealthLatency: TimeInterval?
    var isControlActionRunning = false
    private var isCheckingStatus = false
    /// Whether the onboarding SetupView is currently shown.
    /// Used to suppress duplicate error banners in other views.
    var isOnboarding = false
    var hasCompletedSetup: Bool {
        UserDefaults.standard.bool(forKey: Keys.setupCompleted)
    }
    var isManagedServerRunning: Bool {
        serverProcess?.isRunning == true
    }

    private var serverProcess: Process?
    private var healthTask: Task<Void, Never>?
    private var setupProcess: Process?
    private var didRequestManualStop = false
    private var restartAttempts = 0

    private let inferenceService: InferenceService
    private let log = AppLogger.shared

    private enum Keys {
        static let setupCompleted = "engine.setupCompleted"
        static let engineDirOverride = "engine.directoryOverride"
    }

    init(inferenceService: InferenceService) {
        self.inferenceService = inferenceService
    }

    // MARK: - Engine Directory Resolution

    /// The project source root, derived at compile time from the location of this source file.
    /// This ensures the engine directory is found even when running from Xcode's DerivedData.
    private nonisolated static let compileTimeSourceRoot: URL = {
        // #filePath → .../Auralux/Services/EngineService.swift
        // Walk up 3 levels: Services → Auralux → project root
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<3 { url = url.deletingLastPathComponent() }
        return url
    }()

    /// Locates the AuraluxEngine directory by checking several candidates.
    nonisolated var engineDirectory: URL? {
        // Allow explicit override stored in UserDefaults
        if let override = UserDefaults.standard.string(forKey: Keys.engineDirOverride) {
            let url = URL(fileURLWithPath: override)
            if FileManager.default.fileExists(atPath: url.appendingPathComponent("server.py").path) {
                return url
            }
        }

        let fm = FileManager.default
        let candidates = [
            // Compile-time source root (works from Xcode DerivedData)
            Self.compileTimeSourceRoot
                .appendingPathComponent("AuraluxEngine"),
            // Next to the running executable (SPM .build/debug/)
            Bundle.main.executableURL?
                .deletingLastPathComponent()
                .appendingPathComponent("AuraluxEngine"),
            // CWD (when running from terminal with project root as CWD)
            URL(fileURLWithPath: fm.currentDirectoryPath)
                .appendingPathComponent("AuraluxEngine"),
            // Two levels up from executable (common in .app bundles)
            Bundle.main.bundleURL
                .deletingLastPathComponent()
                .appendingPathComponent("AuraluxEngine"),
            // Inside app bundle Resources
            Bundle.main.resourceURL?
                .appendingPathComponent("AuraluxEngine"),
        ].compactMap { $0 }

        return candidates.first { fm.fileExists(atPath: $0.appendingPathComponent("server.py").path) }
    }

    var isVenvReady: Bool {
        guard let dir = engineDirectory else { return false }
        let venvPython = dir
            .appendingPathComponent("ACE-Step-1.5")
            .appendingPathComponent(".venv")
            .appendingPathComponent("bin")
            .appendingPathComponent("python")
        return FileManager.default.fileExists(atPath: venvPython.path)
    }

    var isACEStepCloned: Bool {
        guard let dir = engineDirectory else { return false }
        let pyproject = dir
            .appendingPathComponent("ACE-Step-1.5")
            .appendingPathComponent("pyproject.toml")
        return FileManager.default.fileExists(atPath: pyproject.path)
    }

    // MARK: - Status Check

    func checkStatus() async {
        guard !isCheckingStatus else { return }
        isCheckingStatus = true
        defer { isCheckingStatus = false }

        guard let dir = engineDirectory else {
            log.warning("AuraluxEngine directory not found", category: .engine)
            state = .error("AuraluxEngine directory not found.")
            return
        }

        log.info("Engine directory: \(dir.path)", category: .engine)

        if await inferenceService.isHealthy() {
            log.info("Server already healthy", category: .engine)
            let health = await inferenceService.fetchHealth()
            if let health, health.modelLoaded {
                updateModelStatus(from: health)
                state = .ready
            } else {
                state = .running
            }
            startHealthMonitoring()
            return
        }

        if !isACEStepCloned || !isVenvReady {
            log.info("ACE-Step not cloned or venv not ready — needs setup", category: .engine)
            state = .notSetup
            return
        }

        log.info("Engine is configured but not running", category: .engine)
        state = .stopped
    }

    // MARK: - Setup

    func runSetup() async {
        guard let engineDir = engineDirectory else {
            state = .error("AuraluxEngine directory not found.")
            return
        }

        let setupScript = engineDir.appendingPathComponent("setup_env.sh")
        guard FileManager.default.fileExists(atPath: setupScript.path) else {
            state = .error("setup_env.sh not found in AuraluxEngine directory.")
            return
        }

        log.info("Running engine setup from \(setupScript.path)", category: .engine)
        state = .settingUp(progress: "Starting environment setup...")
        setupLog = []
        appendLog("Starting Auralux Engine setup...")

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/bin/zsh")
        proc.arguments = ["-l", setupScript.path]
        proc.currentDirectoryURL = engineDir

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        proc.standardOutput = stdoutPipe
        proc.standardError = stderrPipe

        setupProcess = proc

        // Stream stdout
        let stdoutHandle = stdoutPipe.fileHandleForReading
        stdoutHandle.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let line = String(data: data, encoding: .utf8) else { return }
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                Task { @MainActor [weak self] in
                    self?.appendLog(trimmed)
                    self?.updateSetupProgress(trimmed)
                }
            }
        }

        // Stream stderr
        let stderrHandle = stderrPipe.fileHandleForReading
        stderrHandle.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let line = String(data: data, encoding: .utf8) else { return }
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                Task { @MainActor [weak self] in
                    self?.appendLog(trimmed)
                }
            }
        }

        do {
            try proc.run()
        } catch {
            state = .error("Failed to start setup: \(error.localizedDescription)")
            return
        }

        // Wait for process to complete in background
        await withCheckedContinuation { continuation in
            proc.terminationHandler = { _ in
                continuation.resume()
            }
        }

        stdoutHandle.readabilityHandler = nil
        stderrHandle.readabilityHandler = nil
        setupProcess = nil

        if proc.terminationStatus == 0 {
            appendLog("Setup completed successfully!")
            log.info("Engine setup completed successfully", category: .engine)
            UserDefaults.standard.set(true, forKey: Keys.setupCompleted)
            state = .stopped
        } else {
            appendLog("Setup failed with exit code \(proc.terminationStatus)")
            log.error("Engine setup failed with exit code \(proc.terminationStatus)", category: .engine)
            state = .error("Environment setup failed. Check the log for details.")
        }
    }

    private func updateSetupProgress(_ line: String) {
        if line.contains("Cloning") {
            state = .settingUp(progress: "Cloning ACE-Step 1.5 repository...")
        } else if line.contains("Installing uv") {
            state = .settingUp(progress: "Installing uv package manager...")
        } else if line.contains("Syncing Python") || line.contains("uv sync") {
            state = .settingUp(progress: "Installing Python dependencies (this may take several minutes)...")
        } else if line.contains("Environment ready") {
            state = .settingUp(progress: "Environment ready!")
        }
    }

    private func appendLog(_ message: String) {
        setupLog.append(message)
        if setupLog.count > 500 {
            setupLog.removeFirst(100)
        }
        log.debug(message, category: .engine)
    }

    // MARK: - Server Lifecycle

    func startServer() async {
        guard !isControlActionRunning else { return }
        isControlActionRunning = true
        defer { isControlActionRunning = false }
        didRequestManualStop = false
        await startServerInternal()
    }

    private func startServerInternal() async {
        guard let engineDir = engineDirectory else {
            state = .error("AuraluxEngine directory not found.")
            return
        }

        let startScript = engineDir.appendingPathComponent("start_api_server_macos.sh")
        guard FileManager.default.fileExists(atPath: startScript.path) else {
            log.error("start_api_server_macos.sh not found at \(startScript.path)", category: .engine)
            state = .error("start_api_server_macos.sh not found.")
            return
        }

        if await inferenceService.isHealthy() {
            log.info("Server already running — skipping launch", category: .engine)
            state = .running
            startHealthMonitoring()
            await refreshModelStatus()
            return
        }

        log.info("Starting inference server from \(startScript.path)", category: .engine)
        state = .starting
        appendLog("Starting inference server...")

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/bin/zsh")
        proc.arguments = ["-l", startScript.path]
        proc.currentDirectoryURL = engineDir

        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = pipe

        // Log server output
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                Task { @MainActor [weak self] in
                    self?.appendLog(trimmed)
                }
            }
        }

        do {
            try proc.run()
            serverProcess = proc
        } catch {
            state = .error("Failed to start server: \(error.localizedDescription)")
            return
        }

        let maxAttempts = 600 // 5 minutes for cold Python imports and environment startup
        for attempt in 0..<maxAttempts {
            if await inferenceService.isHealthy() {
                log.info("Server healthy after ~\(attempt / 2)s", category: .engine)
                state = .running
                startHealthMonitoring()
                await refreshModelStatus()
                appendLog("Server is running and healthy.")
                return
            }

            if !proc.isRunning {
                log.error("Server process exited with code \(proc.terminationStatus)", category: .engine)
                state = .error("Server process exited unexpectedly (code \(proc.terminationStatus)).")
                return
            }

            if attempt % 10 == 0 && attempt > 0 {
                appendLog("Waiting for server to become ready... (\(attempt / 2)s)")
            }
            try? await Task.sleep(for: .milliseconds(500))
        }

        log.error("Server did not become healthy within 5 minutes", category: .engine)
        state = .error("Server did not become healthy within 5 minutes.")
    }

    func stopServer(allowExternalStop: Bool = true) {
        didRequestManualStop = true
        restartAttempts = 0
        healthTask?.cancel()
        healthTask = nil

        if let proc = serverProcess, proc.isRunning {
            proc.terminate()
            appendLog("Server stopping…")
            // Escalate to SIGKILL after 5 s if the process hasn't exited.
            // Done off the main thread so we don't block the UI.
            let pid = proc.processIdentifier
            Task.detached {
                var waited = 0.0
                while proc.isRunning && waited < 5.0 {
                    try? await Task.sleep(for: .milliseconds(100))
                    waited += 0.1
                }
                if proc.isRunning {
                    kill(pid, SIGKILL)
                }
            }
        } else if allowExternalStop, let pid = runtimeStats.pid, pid > 1 {
            let killProcess = Process()
            killProcess.executableURL = URL(fileURLWithPath: "/bin/kill")
            killProcess.arguments = ["-TERM", "\(pid)"]
            do {
                try killProcess.run()
                appendLog("Sent stop signal to inference server pid \(pid).")
            } catch {
                appendLog("Failed to stop inference server pid \(pid): \(error.localizedDescription)")
            }
        }
        serverProcess = nil
        runtimeStats = .empty
        lastHealthCheckAt = nil
        lastHealthLatency = nil
        state = .stopped
    }

    func restartServer() async {
        guard !isControlActionRunning else { return }
        isControlActionRunning = true
        stopServer(allowExternalStop: true)
        state = .starting
        appendLog("Restarting inference server...")
        try? await Task.sleep(for: .milliseconds(700))
        await startServer()
        isControlActionRunning = false
    }

    // MARK: - Health Monitoring

    private func startHealthMonitoring() {
        healthTask?.cancel()
        healthTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(10))
                guard !Task.isCancelled else { break }
                await self?.performHealthCheck()
            }
        }
    }

    private func performHealthCheck() async {
        let startedAt = Date()
        if let health = await inferenceService.fetchHealth() {
            lastHealthCheckAt = .now
            lastHealthLatency = Date().timeIntervalSince(startedAt)
            restartAttempts = 0
            updateModelStatus(from: health)
            state = health.modelLoaded ? .ready : .running
        } else {
            // Server might have crashed — check if process is still alive
            if let proc = serverProcess, !proc.isRunning {
                guard !didRequestManualStop else {
                    state = .stopped
                    return
                }
                let maxRestarts = 3
                guard restartAttempts < maxRestarts else {
                    log.error("Server crashed \(maxRestarts) times without recovering — giving up", category: .engine)
                    state = .error("Server crashed \(maxRestarts) times. Check logs and restart manually.")
                    return
                }
                restartAttempts += 1
                // 1st failure → 1 s, 2nd → 4 s, 3rd → 16 s then give up.
                let backoffSeconds = pow(4.0, Double(restartAttempts - 1))
                appendLog("Server process died (attempt \(restartAttempts)/\(maxRestarts)). Restarting in \(Int(backoffSeconds))s…")
                try? await Task.sleep(for: .seconds(backoffSeconds))
                await startServerInternal()
            } else if serverProcess != nil {
                appendLog("Inference server stopped responding.")
                state = .error("Inference server stopped responding.")
            } else if serverProcess == nil, !didRequestManualStop {
                // External server went away
                state = .error("Lost connection to inference server.")
            }
        }
    }

    // MARK: - Model Status

    func refreshModelStatus() async {
        await performHealthCheck()
    }

    func refreshNow() async {
        if state.isRunning || state == .starting || serverProcess != nil {
            await performHealthCheck()
        } else {
            await checkStatus()
        }
    }

    func prepareForGeneration() async throws {
        if state == .unknown {
            await checkStatus()
        }

        if state.isReady || state.isRunning {
            return
        }

        if state.needsSetup {
            isOnboarding = true
            throw EnginePreparationError.setupRequired
        }

        guard !state.isBusy, !isControlActionRunning else {
            throw EnginePreparationError.busy
        }

        await startServer()

        if state.isReady || state.isRunning {
            return
        }

        if case .error(let message) = state {
            throw EnginePreparationError.startupFailed(message)
        }

        throw EnginePreparationError.startupFailed("Unable to start the inference server.")
    }

    func triggerModelDownload() async {
        do {
            try await inferenceService.triggerModelDownload()
            appendLog("Model download triggered.")
            state = .settingUp(progress: "Downloading models…")
            // Poll up to 30 min; surface progress via state so the UI can show it.
            let maxAttempts = 1800
            for attempt in 0..<maxAttempts {
                // Propagate cancellation — do not swallow with try?
                try await Task.sleep(for: .seconds(1))
                if let health = await inferenceService.fetchHealth() {
                    if health.modelLoaded {
                        updateModelStatus(from: health)
                        state = .ready
                        appendLog("Models loaded successfully.")
                        return
                    }
                    // Rough progress estimate: ramp 0→80% over 15 min, hold at 80% after that.
                    let pct = min(80, Int(Double(attempt) / 900.0 * 80.0))
                    state = .settingUp(progress: "Downloading models — \(pct)%")
                }
            }
            appendLog("Model download timed out.")
            state = .error("Model download did not complete within 30 minutes.")
        } catch is CancellationError {
            // Task cancelled (e.g. app quit during download). Don't surface as an error.
            if state != .ready { state = .running }
        } catch {
            appendLog("Model download failed: \(error.localizedDescription)")
            state = .error("Model download failed: \(error.localizedDescription)")
        }
    }

    private func updateModelStatus(from health: HealthResponse) {
        modelStatus = ModelStatus(
            ditLoaded: health.modelLoaded,
            llmLoaded: health.llmLoaded ?? false,
            device: health.device ?? "unknown",
            engine: health.engine ?? "unknown",
            ditModel: health.ditModel ?? "",
            llmModel: health.llmModel ?? "",
            error: health.modelError
        )
        runtimeStats = health.stats ?? .empty
    }

    // MARK: - Convenience

    /// Called when the app is about to quit.
    func shutdown() {
        stopServer(allowExternalStop: false)
    }

    /// Attempts to connect to an already-running external server
    /// (useful when the user starts the server manually).
    func connectToExternalServer() async {
        if await inferenceService.isHealthy() {
            didRequestManualStop = false
            state = .running
            startHealthMonitoring()
            await refreshModelStatus()
            appendLog("Connected to external server.")
        } else {
            state = .error("No server found at \(AppConstants.inferenceBaseURL.absoluteString)")
        }
    }
}

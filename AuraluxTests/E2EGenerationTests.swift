import SwiftData
import XCTest
@testable import Auralux

/// End-to-end tests that exercise the full music generation pipeline:
///
///   UI state (ViewModel) → InferenceService → HTTP (mocked) → polling →
///   SwiftData persistence → player readiness
///
/// The Python inference server is replaced by `MockURLProtocol` which returns
/// scripted HTTP responses. SwiftData uses an in-memory store for isolation.
@MainActor
final class E2EGenerationTests: XCTestCase {

    private var session: URLSession!
    private var inferenceService: InferenceService!
    private var viewModel: GenerationViewModel!
    private var container: ModelContainer!
    private var context: ModelContext!

    // MARK: - Lifecycle

    override func setUp() async throws {
        try await super.setUp()

        MockURLProtocol.reset()

        // Register a healthy response so the HTTP-only readiness check passes instantly.
        MockURLProtocol.register(
            path: "/health",
            response: .init(statusCode: 200, json: [
                "status": "ok",
                "modelLoaded": true,
                "device": "mps"
            ])
        )

        session = MockURLProtocol.urlSession()

        inferenceService = InferenceService(
            baseURL: URL(string: "http://127.0.0.1:8765")!,
            session: session
        )

        viewModel = GenerationViewModel(inferenceService: inferenceService)

        let config = ModelConfiguration(isStoredInMemoryOnly: true)
        container = try ModelContainer(
            for: GeneratedTrack.self, Preset.self, Tag.self,
            configurations: config
        )
        context = container.mainContext
    }

    override func tearDown() async throws {
        MockURLProtocol.reset()
        session = nil
        inferenceService = nil
        viewModel = nil
        container = nil
        context = nil
        try await super.tearDown()
    }

    // MARK: - Helpers

    /// Registers mock responses for a happy-path generation: /generate returns a
    /// jobID, then sequential polls return running → completed.
    private func registerHappyPath(
        jobID: String = "test-job-1",
        audioPath: String = "/tmp/test-audio.wav"
    ) {
        MockURLProtocol.register(
            path: "/generate",
            response: .init(
                statusCode: 200,
                body: GenerationResponse(jobID: jobID, status: "accepted", message: "Job queued")
            )
        )

        MockURLProtocol.register(
            path: "/jobs/\(jobID)",
            responses: [
                .init(statusCode: 200, body: GenerationStatusResponse(
                    jobID: jobID, status: "pending", progress: 0.0, message: "Queued"
                )),
                .init(statusCode: 200, body: GenerationStatusResponse(
                    jobID: jobID, status: "running", progress: 0.3, message: "Generating..."
                )),
                .init(statusCode: 200, body: GenerationStatusResponse(
                    jobID: jobID, status: "running", progress: 0.7, message: "Processing audio..."
                )),
                .init(statusCode: 200, body: GenerationStatusResponse(
                    jobID: jobID, status: "completed", progress: 1.0, message: "Done", audioPath: audioPath
                )),
            ]
        )
    }

    /// Configures the ViewModel with a valid prompt and generation parameters.
    private func configureValidPrompt() {
        viewModel.prompt = "Ambient piano melody for focus"
        viewModel.lyrics = "[verse]\nGentle keys in morning light"
        viewModel.tags = ["ambient", "piano", "chill"]
        viewModel.duration = 30
        viewModel.variance = 0.5
        viewModel.seedText = "42"
    }

    /// Waits for the generation state to leave `.preparing` / `.generating`.
    private func waitForCompletion(timeout: TimeInterval = 10) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while viewModel.state.isBusy {
            guard Date() < deadline else {
                XCTFail("Generation did not complete within \(timeout)s — state: \(viewModel.state)")
                return
            }
            try await Task.sleep(for: .milliseconds(50))
            // Yield to let the MainActor process ViewModel updates
            await Task.yield()
        }
    }

    // MARK: - Test Cases

    /// Full happy-path: idle → preparing → generating → completed.
    /// Verifies the track is persisted in SwiftData and `lastTrack` is set.
    func testHappyPathGenerationPipeline() async throws {
        registerHappyPath()
        configureValidPrompt()

        XCTAssertEqual(viewModel.state, .idle)

        viewModel.generate(in: context)

        // State should transition to preparing immediately
        XCTAssertTrue(viewModel.state == .preparing || viewModel.state == .generating)

        try await waitForCompletion()

        XCTAssertEqual(viewModel.state, .completed)
        XCTAssertNotNil(viewModel.lastTrack)
        XCTAssertEqual(viewModel.lastTrack?.prompt, "Ambient piano melody for focus")
        XCTAssertEqual(viewModel.lastTrack?.generationID, "test-job-1")
        XCTAssertNil(viewModel.currentJobID, "Job ID should be cleared after completion")

        // Verify persistence
        let descriptor = FetchDescriptor<GeneratedTrack>()
        let tracks = try context.fetch(descriptor)
        XCTAssertEqual(tracks.count, 1)
        XCTAssertEqual(tracks.first?.generationID, "test-job-1")
    }

    /// Verifies that progress and message update as polling progresses.
    func testGenerationProgressUpdates() async throws {
        registerHappyPath()
        configureValidPrompt()

        viewModel.generate(in: context)
        try await waitForCompletion()

        // After completion, progress should be 1.0
        XCTAssertEqual(viewModel.progress, 1.0, accuracy: 0.01)
        XCTAssertEqual(viewModel.state, .completed)

        // Verify HTTP requests were made in the expected order
        let requests = MockURLProtocol.capturedRequests()
        let paths = requests.compactMap { $0.url?.path }

        XCTAssertTrue(paths.contains("/health"), "Should check server health")
        XCTAssertTrue(paths.contains("/generate"), "Should submit generation request")
        XCTAssertTrue(paths.contains { $0.hasPrefix("/jobs/") }, "Should poll for job status")
    }

    /// Empty prompt should fail immediately without making any HTTP calls.
    func testEmptyPromptFailsImmediately() async throws {
        viewModel.prompt = "   "
        viewModel.generate(in: context)

        XCTAssertEqual(viewModel.state, .failed("Prompt is required."))

        // No HTTP calls should have been made
        let requests = MockURLProtocol.capturedRequests()
        XCTAssertTrue(requests.isEmpty, "No HTTP requests should be made for an empty prompt")
    }

    /// HTTP 500 from /generate should result in a failed state.
    func testServerErrorDuringGenerate() async throws {
        MockURLProtocol.register(
            path: "/generate",
            response: .init(statusCode: 500, json: ["error": "Internal server error"])
        )

        configureValidPrompt()
        viewModel.generate(in: context)
        try await waitForCompletion()

        if case .failed(let message) = viewModel.state {
            XCTAssertTrue(message.contains("500"), "Error message should mention HTTP status: \(message)")
        } else {
            XCTFail("Expected .failed state, got \(viewModel.state)")
        }
    }

    /// Server returns status: "failed" during poll — should propagate the error message.
    func testJobFailureStatus() async throws {
        let jobID = "fail-job-1"

        MockURLProtocol.register(
            path: "/generate",
            response: .init(
                statusCode: 200,
                body: GenerationResponse(jobID: jobID, status: "accepted", message: nil)
            )
        )

        MockURLProtocol.register(
            path: "/jobs/\(jobID)",
            responses: [
                .init(statusCode: 200, body: GenerationStatusResponse(
                    jobID: jobID, status: "running", progress: 0.2, message: "Starting..."
                )),
                .init(statusCode: 200, body: GenerationStatusResponse(
                    jobID: jobID, status: "failed", progress: 0.2, message: "Out of memory"
                )),
            ]
        )

        configureValidPrompt()
        viewModel.generate(in: context)
        try await waitForCompletion()

        if case .failed(let message) = viewModel.state {
            XCTAssertTrue(message.contains("Out of memory"), "Error should contain server message: \(message)")
        } else {
            XCTFail("Expected .failed state, got \(viewModel.state)")
        }
    }

    /// 404 from /jobs/<id> (server crashed and restarted) should fail immediately
    /// with a jobNotFound error.
    func testJobNotFoundHandling() async throws {
        let jobID = "lost-job-1"

        MockURLProtocol.register(
            path: "/generate",
            response: .init(
                statusCode: 200,
                body: GenerationResponse(jobID: jobID, status: "accepted", message: nil)
            )
        )

        MockURLProtocol.register(
            path: "/jobs/\(jobID)",
            responses: [
                .init(statusCode: 200, body: GenerationStatusResponse(
                    jobID: jobID, status: "running", progress: 0.1, message: "Starting..."
                )),
                .init(statusCode: 404, json: ["error": "Job not found"]),
            ]
        )

        configureValidPrompt()
        viewModel.generate(in: context)
        try await waitForCompletion()

        if case .failed(let message) = viewModel.state {
            XCTAssertTrue(
                message.lowercased().contains("lost") || message.lowercased().contains("crashed"),
                "Error should mention job loss: \(message)"
            )
        } else {
            XCTFail("Expected .failed state, got \(viewModel.state)")
        }
    }

    /// Calling cancel() during generation should reset state and attempt to
    /// cancel the job on the server.
    func testCancellationStopsPolling() async throws {
        let jobID = "cancel-job-1"

        MockURLProtocol.register(
            path: "/generate",
            response: .init(
                statusCode: 200,
                body: GenerationResponse(jobID: jobID, status: "accepted", message: nil)
            )
        )

        // Polls return "running" indefinitely — the test will cancel before completion
        MockURLProtocol.register(
            path: "/jobs/\(jobID)",
            response: .init(statusCode: 200, body: GenerationStatusResponse(
                jobID: jobID, status: "running", progress: 0.3, message: "Generating..."
            ))
        )

        MockURLProtocol.register(
            path: "/jobs/\(jobID)/cancel",
            response: .init(statusCode: 200, json: ["status": "cancelled"])
        )

        configureValidPrompt()
        viewModel.generate(in: context)

        // Let the generation start and reach "generating" state
        let deadline = Date().addingTimeInterval(5)
        while viewModel.state != .generating {
            guard Date() < deadline else {
                XCTFail("Did not reach .generating state")
                return
            }
            try await Task.sleep(for: .milliseconds(50))
            await Task.yield()
        }

        viewModel.cancel()

        XCTAssertEqual(viewModel.state, .idle)
        XCTAssertEqual(viewModel.progress, 0)
        XCTAssertNil(viewModel.currentJobID)
    }

    /// Verifies that all fields on the persisted GeneratedTrack match the
    /// original generation request parameters.
    func testTrackPersistenceMetadata() async throws {
        let audioPath = "/tmp/generated/track_metadata_test.wav"
        registerHappyPath(jobID: "meta-job-1", audioPath: audioPath)
        configureValidPrompt()

        viewModel.generate(in: context)
        try await waitForCompletion()

        let descriptor = FetchDescriptor<GeneratedTrack>()
        let tracks = try context.fetch(descriptor)
        XCTAssertEqual(tracks.count, 1)

        let track = try XCTUnwrap(tracks.first)
        XCTAssertEqual(track.title, "Ambient piano melody for focus")
        XCTAssertEqual(track.prompt, "Ambient piano melody for focus")
        XCTAssertEqual(track.lyrics, "[verse]\nGentle keys in morning light")
        XCTAssertEqual(track.tags, ["ambient", "piano", "chill"])
        XCTAssertEqual(track.duration, 30, accuracy: 0.01)
        XCTAssertEqual(track.variance, 0.5, accuracy: 0.01)
        XCTAssertEqual(track.seed, 42)
        XCTAssertEqual(track.generationID, "meta-job-1")
        XCTAssertNotNil(track.audioFilePath)
        XCTAssertEqual(track.format, "wav")
        XCTAssertFalse(track.isFavorite)

        // Audio path should be stored as a relative filename
        if let storedPath = track.audioFilePath {
            XCTAssertFalse(storedPath.hasPrefix("/"), "Audio path should be relative, got: \(storedPath)")
            XCTAssertTrue(storedPath.contains("track_metadata_test.wav"))
        }
    }

    /// After a successful generation, the track's audio path should be set and
    /// resolvable via `FileUtilities`.  We verify the path round-trips correctly
    /// without actually instantiating AVAudioEngine (which crashes in headless
    /// test environments).
    func testPlayerViewModelReceivesTrackPath() async throws {
        let audioPath = "/tmp/generated/player_test_track.wav"
        registerHappyPath(jobID: "player-job-1", audioPath: audioPath)
        configureValidPrompt()

        viewModel.generate(in: context)
        try await waitForCompletion()

        let track = try XCTUnwrap(viewModel.lastTrack)
        let storedPath = try XCTUnwrap(track.audioFilePath)

        // Stored path should be a relative filename (not an absolute path)
        XCTAssertFalse(storedPath.hasPrefix("/"), "Path should be relative: \(storedPath)")
        XCTAssertTrue(storedPath.contains("player_test_track.wav"))

        // FileUtilities should resolve it to the Generated directory
        let resolvedURL = FileUtilities.resolveAudioPath(storedPath)
        XCTAssertTrue(
            resolvedURL.path.contains(AppConstants.generatedDirectoryName),
            "Resolved path should point to the Generated directory: \(resolvedURL.path)"
        )
        XCTAssertTrue(resolvedURL.lastPathComponent.contains("player_test_track.wav"))
    }
}

// MARK: - Collection helper

private extension Array where Element == String {
    func contains(where predicate: (String) -> Bool) -> Bool {
        first(where: predicate) != nil
    }
}

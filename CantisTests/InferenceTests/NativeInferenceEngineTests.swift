import XCTest
import MLX
@testable import Auralux

/// Unit tests for NativeInferenceEngine — verify state machine and
/// error paths without real model weights.
@MainActor
final class NativeInferenceEngineTests: XCTestCase {

    // MARK: - Initial State

    func testInitialModelStateIsNotDownloaded() {
        let engine = NativeInferenceEngine()
        XCTAssertEqual(engine.modelState, .notDownloaded)
    }

    func testInitialIsGeneratingIsFalse() {
        let engine = NativeInferenceEngine()
        XCTAssertFalse(engine.isGenerating)
    }

    func testInitialIsOnboardingIsFalse() {
        let engine = NativeInferenceEngine()
        XCTAssertFalse(engine.isOnboarding)
    }

    // MARK: - ModelState helpers

    func testModelStateIsReady() {
        XCTAssertTrue(ModelState.ready.isReady)
        XCTAssertFalse(ModelState.notDownloaded.isReady)
        XCTAssertFalse(ModelState.loading.isReady)
        XCTAssertFalse(ModelState.error("test").isReady)
    }

    func testModelStateIsLoading() {
        XCTAssertTrue(ModelState.loading.isLoading)
        XCTAssertFalse(ModelState.ready.isLoading)
        XCTAssertFalse(ModelState.notDownloaded.isLoading)
        XCTAssertFalse(ModelState.error("test").isLoading)
    }

    func testModelStateEquality() {
        XCTAssertEqual(ModelState.ready, .ready)
        XCTAssertEqual(ModelState.notDownloaded, .notDownloaded)
        XCTAssertEqual(ModelState.loading, .loading)
        XCTAssertEqual(ModelState.error("foo"), .error("foo"))
        XCTAssertNotEqual(ModelState.error("foo"), .error("bar"))
        XCTAssertNotEqual(ModelState.ready, .loading)
    }

    // MARK: - Weights detection

    func testWeightsExistPropertyIsAccessible() {
        let engine = NativeInferenceEngine()
        // No weights in the test sandbox — just verify the property is readable.
        let _ = engine.weightsExist
    }

    func testMlxModelDirectoryIsInsideAppSupport() {
        let engine = NativeInferenceEngine()
        let modelDirectory = engine.mlxModelDirectory(for: .turbo)
        XCTAssertTrue(modelDirectory.path.contains("Auralux"))
        XCTAssertTrue(modelDirectory.lastPathComponent == "ace-step-v1.5-mlx")
    }

    // MARK: - Cancellation

    func testCancelGenerationIsNoOpWhenIdle() {
        let engine = NativeInferenceEngine()
        engine.cancelGeneration()
        XCTAssertFalse(engine.isGenerating)
        XCTAssertEqual(engine.modelState, .notDownloaded)
    }

    func testShutdownIsIdempotent() {
        let engine = NativeInferenceEngine()
        engine.shutdown()
        engine.shutdown()
        XCTAssertFalse(engine.isGenerating)
    }

    // MARK: - Generate with unloaded models

    func testGenerateWithUnloadedModelsThrowsImmediately() async {
        let engine = NativeInferenceEngine()
        XCTAssertEqual(engine.modelState, .notDownloaded)

        let stream = engine.generate(request: .default)
        do {
            for try await _ in stream {
                XCTFail("Expected immediate error — no events should be yielded")
            }
            XCTFail("Stream should have thrown NativeEngineError.modelsNotLoaded")
        } catch let error as NativeEngineError {
            if case .modelsNotLoaded = error { return }
            XCTFail("Unexpected NativeEngineError: \(error)")
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    func testGenerateDoesNotSetIsGeneratingWhenModelsNotLoaded() async {
        let engine = NativeInferenceEngine()
        let stream = engine.generate(request: .default)
        do { for try await _ in stream {} } catch {}
        XCTAssertFalse(engine.isGenerating)
    }
}

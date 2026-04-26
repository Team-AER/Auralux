import SwiftData
import XCTest
@testable import Auralux

@MainActor
final class ViewModelTests: XCTestCase {

    // MARK: - GenerationViewModel Tags

    func testGenerationViewModelTagMutations() {
        let viewModel = GenerationViewModel(inferenceService: InferenceService())

        viewModel.addTag("Ambient")
        viewModel.addTag("ambient")
        viewModel.addTag("piano")

        XCTAssertEqual(viewModel.tags, ["lofi", "piano", "chill", "ambient"])

        viewModel.removeTag("ambient")
        XCTAssertEqual(viewModel.tags, ["lofi", "piano", "chill"])
    }

    func testAddEmptyTagIsIgnored() {
        let viewModel = GenerationViewModel(inferenceService: InferenceService())
        viewModel.addTag("")
        viewModel.addTag("   ")
        XCTAssertEqual(viewModel.tags, ["lofi", "piano", "chill"])
    }

    func testRemoveNonexistentTagDoesNothing() {
        let viewModel = GenerationViewModel(inferenceService: InferenceService())
        viewModel.addTag("rock")
        viewModel.removeTag("jazz")
        XCTAssertEqual(viewModel.tags, ["lofi", "piano", "chill", "rock"])
    }

    // MARK: - GenerationViewModel Preset

    func testApplyingPresetUpdatesParameters() {
        let viewModel = GenerationViewModel(inferenceService: InferenceService())
        let preset = Preset(
            name: "Test",
            summary: "Summary",
            prompt: "Prompt",
            lyricTemplate: "Lyrics",
            tags: ["a", "b"],
            duration: 55,
            variance: 0.66
        )

        viewModel.applyPreset(preset)

        XCTAssertEqual(viewModel.prompt, "Prompt")
        XCTAssertEqual(viewModel.lyrics, "Lyrics")
        XCTAssertEqual(viewModel.tags, ["a", "b"])
        XCTAssertEqual(viewModel.duration, 55)
        XCTAssertEqual(viewModel.variance, 0.66)
    }

    // MARK: - GenerationViewModel State

    func testInitialStateIsIdle() {
        let viewModel = GenerationViewModel(inferenceService: InferenceService())
        XCTAssertEqual(viewModel.state, .idle)
        XCTAssertEqual(viewModel.progress, 0)
        XCTAssertNil(viewModel.currentJobID)
        XCTAssertNil(viewModel.lastTrack)
    }

    func testGenerateWithEmptyPromptFails() throws {
        let viewModel = GenerationViewModel(inferenceService: InferenceService())
        viewModel.prompt = "   "

        let config = ModelConfiguration(isStoredInMemoryOnly: true)
        let container = try ModelContainer(for: GeneratedTrack.self, Preset.self, Tag.self, configurations: config)
        let engine = EngineService(inferenceService: InferenceService())
        viewModel.generate(in: container.mainContext, engine: engine)

        XCTAssertEqual(viewModel.state, .failed("Prompt is required."))
    }

    func testCancelResetsState() {
        let viewModel = GenerationViewModel(inferenceService: InferenceService())
        viewModel.prompt = "test song"
        viewModel.cancel()

        XCTAssertEqual(viewModel.state, .idle)
        XCTAssertEqual(viewModel.progress, 0)
        XCTAssertNil(viewModel.currentJobID)
    }

    func testGenerateWithEmptyPromptLeavesTaskNil() throws {
        let viewModel = GenerationViewModel(inferenceService: InferenceService())
        viewModel.prompt = ""

        let config = ModelConfiguration(isStoredInMemoryOnly: true)
        let container = try ModelContainer(for: GeneratedTrack.self, Preset.self, Tag.self, configurations: config)
        let engine = EngineService(inferenceService: InferenceService())
        viewModel.generate(in: container.mainContext, engine: engine)

        // Empty-prompt guard returns before assigning generationTask.
        // Cancelling immediately should not throw or crash.
        viewModel.cancel()
        XCTAssertEqual(viewModel.state, .idle)
    }

    func testSettingsMaxConcurrentJobsClampSavesOnce() {
        let suiteName = "test-clamp-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }

        let vm = SettingsViewModel(defaults: defaults)
        vm.maxConcurrentJobs = 10  // triggers clamping to 4

        XCTAssertEqual(vm.maxConcurrentJobs, 4)
        // The saved value should be 4, not 10.
        XCTAssertEqual(defaults.integer(forKey: "settings.maxConcurrentJobs"), 4)
    }

    // MARK: - GenerationState

    func testGenerationStateIsBusy() {
        XCTAssertFalse(GenerationState.idle.isBusy)
        XCTAssertTrue(GenerationState.preparing.isBusy)
        XCTAssertTrue(GenerationState.generating.isBusy)
        XCTAssertFalse(GenerationState.completed.isBusy)
        XCTAssertFalse(GenerationState.failed("error").isBusy)
    }

    // MARK: - SettingsViewModel

    func testSettingsDefaultValues() {
        let defaults = UserDefaults(suiteName: "test-\(UUID().uuidString)")!
        let vm = SettingsViewModel(defaults: defaults)

        XCTAssertEqual(vm.quantizationMode, .fp16)
        XCTAssertFalse(vm.lowMemoryMode)
        XCTAssertTrue(vm.autoStartServer)
        XCTAssertEqual(vm.maxConcurrentJobs, 1)
        XCTAssertEqual(vm.defaultExportFormat, .wav)
    }

    func testSettingsPersistenceRoundTrip() {
        let suiteName = "test-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!

        let vm1 = SettingsViewModel(defaults: defaults)
        vm1.quantizationMode = .int8
        vm1.lowMemoryMode = true
        vm1.autoStartServer = false
        vm1.maxConcurrentJobs = 3
        vm1.defaultExportFormat = .alac  // .flac is unavailable, use .alac

        let vm2 = SettingsViewModel(defaults: defaults)
        XCTAssertEqual(vm2.quantizationMode, .int8)
        XCTAssertTrue(vm2.lowMemoryMode)
        XCTAssertFalse(vm2.autoStartServer)
        XCTAssertEqual(vm2.maxConcurrentJobs, 3)
        XCTAssertEqual(vm2.defaultExportFormat, .alac)

        defaults.removePersistentDomain(forName: suiteName)
    }

    func testSettingsUnavailableExportFormatFallsBackToWAV() {
        let suiteName = "test-unavailable-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defer { defaults.removePersistentDomain(forName: suiteName) }

        // Manually persist an unavailable format value as if it was saved by an older build.
        defaults.set(AudioExportFormat.flac.rawValue, forKey: "settings.defaultExportFormat")

        let vm = SettingsViewModel(defaults: defaults)
        XCTAssertEqual(vm.defaultExportFormat, .wav, "Unavailable format should fall back to .wav")
    }

    func testSettingsMaxConcurrentJobsClamped() {
        let defaults = UserDefaults(suiteName: "test-\(UUID().uuidString)")!
        let vm = SettingsViewModel(defaults: defaults)

        vm.maxConcurrentJobs = 10
        XCTAssertEqual(vm.maxConcurrentJobs, 4)

        vm.maxConcurrentJobs = 0
        XCTAssertEqual(vm.maxConcurrentJobs, 1)
    }

    func testSettingsResetToDefaults() {
        let defaults = UserDefaults(suiteName: "test-\(UUID().uuidString)")!
        let vm = SettingsViewModel(defaults: defaults)
        vm.quantizationMode = .int8
        vm.lowMemoryMode = true
        vm.maxConcurrentJobs = 4

        vm.resetToDefaults()

        XCTAssertEqual(vm.quantizationMode, .fp16)
        XCTAssertFalse(vm.lowMemoryMode)
        XCTAssertEqual(vm.maxConcurrentJobs, 1)
    }

    // MARK: - PlayerViewModel

    func testPlayerViewModelInitialState() {
        let vm = PlayerViewModel()
        XCTAssertNil(vm.loadedPath)
        XCTAssertNil(vm.errorMessage)
        XCTAssertFalse(vm.isPlaying)
        XCTAssertEqual(vm.progress, 0)
        XCTAssertTrue(vm.waveformSamples.isEmpty)
    }

    func testPlayerViewModelLoadInvalidPath() {
        let vm = PlayerViewModel()
        vm.load(path: "/nonexistent/audio.wav")
        XCTAssertNotNil(vm.errorMessage)
    }

    func testPlayerViewModelLoadNilPath() {
        let vm = PlayerViewModel()
        vm.load(path: nil)
        XCTAssertNil(vm.loadedPath)
        XCTAssertNil(vm.errorMessage)
    }

    func testPlayerViewModelClearError() {
        let vm = PlayerViewModel()
        vm.load(path: "/nonexistent/audio.wav")
        XCTAssertNotNil(vm.errorMessage)
        vm.clearError()
        XCTAssertNil(vm.errorMessage)
    }

    // MARK: - SidebarViewModel

    func testSidebarViewModelDefaultSection() {
        let vm = SidebarViewModel()
        XCTAssertEqual(vm.selectedSection, .generate)
    }
}

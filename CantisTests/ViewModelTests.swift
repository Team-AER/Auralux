import SwiftData
import XCTest
@testable import Auralux

@MainActor
final class ViewModelTests: XCTestCase {

    // MARK: - GenerationViewModel Tags

    func testGenerationViewModelTagMutations() {
        let viewModel = GenerationViewModel(engine: NativeInferenceEngine())

        viewModel.addTag("Ambient")
        viewModel.addTag("ambient")
        viewModel.addTag("piano")

        XCTAssertEqual(viewModel.tags, ["lofi", "piano", "chill", "ambient"])

        viewModel.removeTag("ambient")
        XCTAssertEqual(viewModel.tags, ["lofi", "piano", "chill"])
    }

    func testAddEmptyTagIsIgnored() {
        let viewModel = GenerationViewModel(engine: NativeInferenceEngine())
        viewModel.addTag("")
        viewModel.addTag("   ")
        XCTAssertEqual(viewModel.tags, ["lofi", "piano", "chill"])
    }

    func testRemoveNonexistentTagDoesNothing() {
        let viewModel = GenerationViewModel(engine: NativeInferenceEngine())
        viewModel.addTag("rock")
        viewModel.removeTag("jazz")
        XCTAssertEqual(viewModel.tags, ["lofi", "piano", "chill", "rock"])
    }

    // MARK: - GenerationViewModel Preset

    func testApplyingPresetUpdatesParameters() {
        let viewModel = GenerationViewModel(engine: NativeInferenceEngine())
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
        let viewModel = GenerationViewModel(engine: NativeInferenceEngine())
        XCTAssertEqual(viewModel.state, .idle)
        XCTAssertEqual(viewModel.progress, 0)
        XCTAssertNil(viewModel.currentJobID)
        XCTAssertNil(viewModel.lastTrack)
    }

    func testGenerateWithEmptyPromptFails() throws {
        let engine = NativeInferenceEngine()
        let viewModel = GenerationViewModel(engine: engine)
        viewModel.prompt = "   "

        let config = ModelConfiguration(isStoredInMemoryOnly: true)
        let container = try ModelContainer(for: GeneratedTrack.self, Preset.self, Tag.self, configurations: config)
        viewModel.generate(in: container.mainContext)

        XCTAssertEqual(viewModel.state, .failed("Prompt is required."))
    }

    func testCancelResetsState() {
        let viewModel = GenerationViewModel(engine: NativeInferenceEngine())
        viewModel.prompt = "test song"
        viewModel.cancel()

        XCTAssertEqual(viewModel.state, .idle)
        XCTAssertEqual(viewModel.progress, 0)
        XCTAssertNil(viewModel.currentJobID)
    }

    func testGenerateWithEmptyPromptLeavesTaskNil() throws {
        let engine = NativeInferenceEngine()
        let viewModel = GenerationViewModel(engine: engine)
        viewModel.prompt = ""

        let config = ModelConfiguration(isStoredInMemoryOnly: true)
        let container = try ModelContainer(for: GeneratedTrack.self, Preset.self, Tag.self, configurations: config)
        viewModel.generate(in: container.mainContext)

        // Empty-prompt guard returns before assigning generationTask.
        // Cancelling immediately should not throw or crash.
        viewModel.cancel()
        XCTAssertEqual(viewModel.state, .idle)
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
        // First-launch default tracks the machine class: ≤ 16 GiB Macs opt in.
        XCTAssertEqual(vm.lowMemoryMode, AppConstants.isLowMemoryMachine)
        XCTAssertEqual(vm.defaultExportFormat, .wav)
    }

    func testSettingsPersistenceRoundTrip() {
        let suiteName = "test-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!

        let vm1 = SettingsViewModel(defaults: defaults)
        vm1.quantizationMode = .fp16
        vm1.lowMemoryMode = true
        vm1.defaultExportFormat = .alac  // .flac is unavailable, use .alac

        let vm2 = SettingsViewModel(defaults: defaults)
        XCTAssertEqual(vm2.quantizationMode, .fp16)
        XCTAssertTrue(vm2.lowMemoryMode)
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

    func testSettingsResetToDefaults() {
        let defaults = UserDefaults(suiteName: "test-\(UUID().uuidString)")!
        let vm = SettingsViewModel(defaults: defaults)
        vm.quantizationMode = .fp16
        vm.lowMemoryMode = !AppConstants.isLowMemoryMachine  // flip away from the machine default so reset is observable

        vm.resetToDefaults()

        XCTAssertEqual(vm.quantizationMode, .fp16)
        XCTAssertEqual(vm.lowMemoryMode, AppConstants.isLowMemoryMachine)
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

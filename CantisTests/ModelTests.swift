import XCTest
@testable import Auralux

final class ModelTests: XCTestCase {

    // MARK: - GenerationParameters

    func testGenerationParametersCodableRoundTrip() throws {
        let params = GenerationParameters(
            prompt: "ambient piano",
            lyrics: "[verse]",
            tags: ["ambient", "piano"],
            duration: 48,
            variance: 0.3,
            seed: 42
        )

        let data = try JSONEncoder().encode(params)
        let decoded = try JSONDecoder().decode(GenerationParameters.self, from: data)
        XCTAssertEqual(params, decoded)
    }

    func testGenerationParametersDefaultValues() {
        let defaults = GenerationParameters.default
        XCTAssertEqual(defaults.prompt, "chill lofi piano")
        XCTAssertTrue(defaults.lyrics.contains("[verse]"))
        XCTAssertEqual(defaults.tags, ["lofi", "piano", "chill"])
        XCTAssertEqual(defaults.duration, 30)
        XCTAssertEqual(defaults.variance, 0.5)
        XCTAssertNil(defaults.seed)
    }

    func testGenerationParametersNilSeedEncoding() throws {
        let params = GenerationParameters(prompt: "test", lyrics: "", tags: [], duration: 10, variance: 0.5, seed: nil)
        let data = try JSONEncoder().encode(params)
        let decoded = try JSONDecoder().decode(GenerationParameters.self, from: data)
        XCTAssertNil(decoded.seed)
    }

    func testGenerationParametersHashable() {
        let a = GenerationParameters(prompt: "x", lyrics: "", tags: ["a"], duration: 30, variance: 0.5, seed: 1)
        let b = GenerationParameters(prompt: "x", lyrics: "", tags: ["a"], duration: 30, variance: 0.5, seed: 1)
        let c = GenerationParameters(prompt: "y", lyrics: "", tags: ["a"], duration: 30, variance: 0.5, seed: 1)
        XCTAssertEqual(a, b)
        XCTAssertNotEqual(a, c)
        XCTAssertEqual(a.hashValue, b.hashValue)
    }

    // MARK: - GeneratedTrack

    func testGeneratedTrackDefaultValues() {
        let track = GeneratedTrack(
            title: "Test",
            prompt: "ambient",
            lyrics: "",
            tags: ["ambient"],
            duration: 30,
            variance: 0.5,
            seed: nil,
            generationID: "abc-123"
        )

        XCTAssertEqual(track.title, "Test")
        XCTAssertEqual(track.format, "wav")
        XCTAssertFalse(track.isFavorite)
        XCTAssertNil(track.audioFilePath)
        XCTAssertNotNil(track.createdAt)
    }

    func testGeneratedTrackCustomFormat() {
        let track = GeneratedTrack(
            title: "Song",
            prompt: "rock",
            lyrics: "[chorus]",
            tags: ["rock"],
            duration: 60,
            variance: 0.8,
            seed: 42,
            generationID: "def-456",
            audioFilePath: "/tmp/test.flac",
            format: "flac",
            isFavorite: true
        )

        XCTAssertEqual(track.format, "flac")
        XCTAssertTrue(track.isFavorite)
        XCTAssertEqual(track.audioFilePath, "/tmp/test.flac")
        XCTAssertEqual(track.seed, 42)
    }

    // MARK: - Preset

    func testPresetParametersComputed() {
        let preset = Preset(
            name: "Lo-Fi",
            summary: "Chill beats",
            prompt: "lofi hip hop",
            lyricTemplate: "[verse]\nRelax...",
            tags: ["lofi", "chill"],
            duration: 45,
            variance: 0.3
        )

        let params = preset.parameters
        XCTAssertEqual(params.prompt, "lofi hip hop")
        XCTAssertEqual(params.lyrics, "[verse]\nRelax...")
        XCTAssertEqual(params.tags, ["lofi", "chill"])
        XCTAssertEqual(params.duration, 45)
        XCTAssertEqual(params.variance, 0.3)
        XCTAssertNil(params.seed)
    }

    // MARK: - Tag

    func testTagDefaultCategory() {
        let tag = Tag(name: "ambient")
        XCTAssertEqual(tag.category, "custom")
        XCTAssertEqual(tag.name, "ambient")
    }

    func testTagCustomCategory() {
        let tag = Tag(name: "piano", category: "instrument")
        XCTAssertEqual(tag.category, "instrument")
    }

    // MARK: - AudioExportFormat

    func testAudioExportFormatFileExtension() {
        XCTAssertEqual(AudioExportFormat.wav.fileExtension, "wav")
        XCTAssertEqual(AudioExportFormat.flac.fileExtension, "flac")
        XCTAssertEqual(AudioExportFormat.mp3.fileExtension, "mp3")
        // AAC and ALAC use the MP4 container (.m4a) for compatibility with
        // macOS/iOS media apps; the raw codec extension (.aac/.alac) is not
        // a valid standalone container.
        XCTAssertEqual(AudioExportFormat.aac.fileExtension, "m4a")
        XCTAssertEqual(AudioExportFormat.alac.fileExtension, "m4a")
    }

    func testAudioExportFormatCodable() throws {
        let config = AudioExportConfiguration(
            format: .flac,
            sampleRate: 48000,
            title: "My Track",
            tags: ["ambient"]
        )
        let data = try JSONEncoder().encode(config)
        let decoded = try JSONDecoder().decode(AudioExportConfiguration.self, from: data)
        XCTAssertEqual(decoded.format, .flac)
        XCTAssertEqual(decoded.sampleRate, 48000)
        XCTAssertEqual(decoded.title, "My Track")
    }

    // MARK: - ModelArtifact

    func testModelArtifactIdentifiable() {
        let artifact = ModelArtifact(
            name: "acestep-v15-turbo",
            repoID: "ACE-Step/Ace-Step1.5",
            description: "DiT turbo model",
            estimatedSizeGB: 2.5
        )
        XCTAssertEqual(artifact.id, "acestep-v15-turbo")
    }

    func testModelArtifactCodable() throws {
        let artifact = ModelArtifact(
            name: "acestep-5Hz-lm-0.6B",
            repoID: "ACE-Step/acestep-5Hz-lm-0.6B",
            description: "5Hz LM model",
            estimatedSizeGB: 1.2
        )
        let data = try JSONEncoder().encode(artifact)
        let decoded = try JSONDecoder().decode(ModelArtifact.self, from: data)
        XCTAssertEqual(decoded.name, artifact.name)
        XCTAssertEqual(decoded.repoID, artifact.repoID)
        XCTAssertEqual(decoded.estimatedSizeGB, artifact.estimatedSizeGB)
    }

    func testKnownArtifactsNotEmpty() {
        XCTAssertFalse(ModelManagerService.knownArtifacts.isEmpty)
        XCTAssertTrue(ModelManagerService.knownArtifacts.contains { $0.name == "ace-step-v1.5-turbo-mlx" })
    }
}

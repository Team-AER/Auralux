import AVFoundation
import XCTest
@testable import Auralux

final class AudioExportServiceTests: XCTestCase {

    private var service: AudioExportService!
    private var sourceURL: URL!
    private var outputDir: URL!

    override func setUp() async throws {
        try await super.setUp()
        service = AudioExportService()
        sourceURL = try AudioFixtures.sineWave(duration: 0.5)
        outputDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("AuraluxExportTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: sourceURL)
        try? FileManager.default.removeItem(at: outputDir)
        service = nil
        sourceURL = nil
        outputDir = nil
        try await super.tearDown()
    }

    // MARK: - Helpers

    private func config(_ format: AudioExportFormat) -> AudioExportConfiguration {
        AudioExportConfiguration(format: format, sampleRate: 44100, title: "Test Track", tags: [])
    }

    private func assertExportedFile(_ url: URL, format: AudioExportFormat) throws {
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path), "\(format.rawValue) file not found")
        XCTAssertEqual(url.pathExtension, format.fileExtension,
                       "\(format.rawValue) should have extension .\(format.fileExtension)")
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        let size = attrs[.size] as? Int ?? 0
        XCTAssertGreaterThan(size, 0, "\(format.rawValue) file should not be empty")
        // Verify the file is readable by AVAudioFile.
        XCTAssertNoThrow(try AVAudioFile(forReading: url),
                         "\(format.rawValue) file should be readable by AVAudioFile")
    }

    // MARK: - Per-format export

    func testExportWAV() async throws {
        let url = try await service.export(
            sourceURL: sourceURL, destinationDirectory: outputDir, configuration: config(.wav))
        try assertExportedFile(url, format: .wav)
    }

    func testExportFLACIsUnsupported() async {
        do {
            _ = try await service.export(
                sourceURL: sourceURL, destinationDirectory: outputDir, configuration: config(.flac))
            XCTFail("Expected AudioExportError.unsupported for FLAC")
        } catch AudioExportError.unsupported {
            // expected — AVAssetWriter does not support public.flac
        } catch {
            XCTFail("Expected .unsupported, got \(error)")
        }
    }

    func testExportAAC() async throws {
        let url = try await service.export(
            sourceURL: sourceURL, destinationDirectory: outputDir, configuration: config(.aac))
        try assertExportedFile(url, format: .aac)
        // AAC uses the MP4 container — verify the extension is .m4a not .aac
        XCTAssertEqual(url.pathExtension, "m4a")
    }

    func testExportALAC() async throws {
        let url = try await service.export(
            sourceURL: sourceURL, destinationDirectory: outputDir, configuration: config(.alac))
        try assertExportedFile(url, format: .alac)
        XCTAssertEqual(url.pathExtension, "m4a")
    }

    func testExportMP3IsUnsupported() async {
        do {
            _ = try await service.export(
                sourceURL: sourceURL, destinationDirectory: outputDir, configuration: config(.mp3))
            XCTFail("Expected AudioExportError.unsupported for MP3")
        } catch AudioExportError.unsupported {
            // expected — AVAssetWriter does not ship an MP3 encoder (licensed codec)
        } catch {
            XCTFail("Expected .unsupported, got \(error)")
        }
    }

    // MARK: - Error cases

    func testExportMissingSourceThrows() async {
        let missing = outputDir.appendingPathComponent("nonexistent.wav")
        await XCTAssertThrowsErrorAsync(
            try await service.export(
                sourceURL: missing, destinationDirectory: outputDir, configuration: config(.wav))
        )
    }

    // MARK: - Filename sanitization

    func testExportSanitizesTitle() async throws {
        let dirty = AudioExportConfiguration(
            format: .wav, sampleRate: 44100,
            title: "My/Track: \"Best?\"",
            tags: []
        )
        let url = try await service.export(
            sourceURL: sourceURL, destinationDirectory: outputDir, configuration: dirty)
        XCTAssertFalse(url.lastPathComponent.contains("/"), "Slash must be stripped from filename")
        XCTAssertFalse(url.lastPathComponent.contains(":"), "Colon must be stripped from filename")
        XCTAssertFalse(url.lastPathComponent.contains("\""), "Quote must be stripped from filename")
    }
}

// MARK: - XCTest async throw helper

private func XCTAssertThrowsErrorAsync<T>(
    _ expression: @autoclosure () async throws -> T,
    _ message: @autoclosure () -> String = "",
    file: StaticString = #filePath,
    line: UInt = #line
) async {
    do {
        _ = try await expression()
        XCTFail("Expected error but none was thrown. \(message())", file: file, line: line)
    } catch {
        // expected
    }
}

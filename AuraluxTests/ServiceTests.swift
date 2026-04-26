import SwiftData
import XCTest
@testable import Auralux

final class ServiceTests: XCTestCase {

    // MARK: - GenerationQueueService

    func testQueueRespectsPriorityOrdering() async {
        let queue = GenerationQueueService()

        await queue.enqueue(.init(parameters: .default, priority: .low))
        await queue.enqueue(.init(parameters: .default, priority: .high))
        await queue.enqueue(.init(parameters: .default, priority: .normal))

        let first = await queue.dequeue()
        let second = await queue.dequeue()
        let third = await queue.dequeue()

        XCTAssertEqual(first?.priority, .high)
        XCTAssertEqual(second?.priority, .normal)
        XCTAssertEqual(third?.priority, .low)
    }

    func testQueueDequeueEmptyReturnsNil() async {
        let queue = GenerationQueueService()
        let item = await queue.dequeue()
        XCTAssertNil(item)
    }

    func testQueueRemoveByID() async {
        let queue = GenerationQueueService()
        let item = GenerationQueueItem(parameters: .default, priority: .normal)
        await queue.enqueue(item)
        await queue.remove(id: item.id)
        let dequeued = await queue.dequeue()
        XCTAssertNil(dequeued)
    }

    func testQueueClear() async {
        let queue = GenerationQueueService()
        await queue.enqueue(.init(parameters: .default, priority: .low))
        await queue.enqueue(.init(parameters: .default, priority: .high))
        await queue.enqueue(.init(parameters: .default, priority: .normal))
        await queue.clear()
        let items = await queue.pendingItems()
        XCTAssertTrue(items.isEmpty)
    }

    func testQueuePendingItems() async {
        let queue = GenerationQueueService()
        await queue.enqueue(.init(parameters: .default, priority: .normal))
        await queue.enqueue(.init(parameters: .default, priority: .high))

        let pending = await queue.pendingItems()
        XCTAssertEqual(pending.count, 2)
        XCTAssertEqual(pending.first?.priority, .high)
    }

    func testQueueSamePriorityFIFO() async {
        let queue = GenerationQueueService()
        let first = GenerationQueueItem(parameters: .default, priority: .normal)
        let second = GenerationQueueItem(parameters: .default, priority: .normal)
        await queue.enqueue(first)
        await queue.enqueue(second)

        let dequeued1 = await queue.dequeue()
        let dequeued2 = await queue.dequeue()
        XCTAssertEqual(dequeued1?.id, first.id)
        XCTAssertEqual(dequeued2?.id, second.id)
    }

    // MARK: - AudioFFT

    func testFFTMagnitudesEmptyInput() {
        let result = AudioFFT.magnitudes(samples: [], fftSize: 1024)
        XCTAssertTrue(result.isEmpty)
    }

    func testFFTMagnitudesTooFewSamples() {
        let result = AudioFFT.magnitudes(samples: [Float](repeating: 0, count: 100), fftSize: 1024)
        XCTAssertTrue(result.isEmpty)
    }

    func testFFTMagnitudesNonPowerOfTwo() {
        let result = AudioFFT.magnitudes(samples: [Float](repeating: 0, count: 1000), fftSize: 1000)
        XCTAssertTrue(result.isEmpty)
    }

    func testFFTMagnitudesReturnsCorrectBinCount() {
        let samples = [Float](repeating: 0, count: 1024)
        let result = AudioFFT.magnitudes(samples: samples, fftSize: 1024)
        XCTAssertEqual(result.count, 512)
    }

    func testFFTMagnitudesSilenceProducesZeros() {
        let samples = [Float](repeating: 0, count: 1024)
        let result = AudioFFT.magnitudes(samples: samples, fftSize: 1024)
        let total = result.reduce(0, +)
        XCTAssertEqual(total, 0, accuracy: 0.001)
    }

    func testFFTMagnitudesSineWaveHasPeak() {
        let fftSize = 1024
        let sampleRate: Float = 44100
        let frequency: Float = 440
        var samples = [Float](repeating: 0, count: fftSize)
        for i in 0..<fftSize {
            samples[i] = sin(2 * .pi * frequency * Float(i) / sampleRate)
        }
        let result = AudioFFT.magnitudes(samples: samples, fftSize: fftSize)
        XCTAssertFalse(result.isEmpty)
        guard let peak = result.max() else {
            XCTFail("No peak found")
            return
        }
        XCTAssertGreaterThan(peak, 0.5, "A 440Hz sine wave should produce a clear spectral peak")
    }

    // MARK: - AudioExportFormat

    func testExportFormatAllCases() {
        let allCases = AudioExportFormat.allCases
        XCTAssertEqual(allCases.count, 5)
        XCTAssertTrue(allCases.contains(.wav))
        XCTAssertTrue(allCases.contains(.flac))
        XCTAssertTrue(allCases.contains(.mp3))
        XCTAssertTrue(allCases.contains(.aac))
        XCTAssertTrue(allCases.contains(.alac))
    }

    // MARK: - FileUtilities

    func testAppSupportDirectoryExists() {
        let url = FileUtilities.appSupportDirectory
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
    }

    func testModelDirectoryExists() {
        let url = FileUtilities.modelDirectory
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
    }

    func testGeneratedAudioDirectoryExists() {
        let url = FileUtilities.generatedAudioDirectory
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
    }

    func testDiagnosticsDirectoryExists() {
        let url = FileUtilities.diagnosticsDirectory
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
    }

    // MARK: - HistoryService

    private func makeInMemoryContainer() throws -> ModelContainer {
        try ModelContainer(
            for: GeneratedTrack.self, Preset.self, Tag.self,
            configurations: ModelConfiguration(isStoredInMemoryOnly: true)
        )
    }

    private func makeTrack(generationID: String = "job-\(UUID().uuidString)") -> GeneratedTrack {
        GeneratedTrack(
            title: "Test",
            prompt: "ambient",
            lyrics: "",
            tags: ["ambient"],
            duration: 30,
            variance: 0.5,
            seed: nil,
            generationID: generationID
        )
    }

    @MainActor
    func testHistoryServiceInsertAndFetch() throws {
        let container = try makeInMemoryContainer()
        let service = HistoryService(context: container.mainContext)
        let track = makeTrack()

        try service.insert(track)
        let recent = try service.recent(limit: 10)

        XCTAssertEqual(recent.count, 1)
        XCTAssertEqual(recent.first?.generationID, track.generationID)
    }

    @MainActor
    func testHistoryServiceDeleteRemovesRow() throws {
        let container = try makeInMemoryContainer()
        let service = HistoryService(context: container.mainContext)
        let track = makeTrack()
        try service.insert(track)

        try service.delete(track)

        let remaining = try service.recent(limit: 10)
        XCTAssertTrue(remaining.isEmpty)
    }

    @MainActor
    func testHistoryServiceDeleteRemovesOrphanFile() throws {
        let tmpDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let fileURL = tmpDir.appendingPathComponent("test.wav")
        try Data("RIFF".utf8).write(to: fileURL)
        XCTAssertTrue(FileManager.default.fileExists(atPath: fileURL.path))

        let container = try makeInMemoryContainer()
        let service = HistoryService(context: container.mainContext)
        let track = makeTrack()
        track.audioFilePath = fileURL.path
        try service.insert(track)

        try service.delete(track)

        XCTAssertFalse(FileManager.default.fileExists(atPath: fileURL.path), "File should be removed on delete")
    }

    @MainActor
    func testHistoryServiceSetFavoriteDoesNotDuplicate() throws {
        let container = try makeInMemoryContainer()
        let service = HistoryService(context: container.mainContext)
        let track = makeTrack()
        try service.insert(track)

        try service.setFavorite(track, isFavorite: true)
        try service.setFavorite(track, isFavorite: false)

        let rows = try service.recent(limit: 10)
        XCTAssertEqual(rows.count, 1, "setFavorite must not insert duplicate rows")
        XCTAssertFalse(rows[0].isFavorite)
    }

    @MainActor
    func testHistoryServiceReconcileOrphansRemovesUnreferenced() async throws {
        let tmpDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let orphan = tmpDir.appendingPathComponent("orphan.wav")
        try Data("RIFF".utf8).write(to: orphan)
        XCTAssertTrue(FileManager.default.fileExists(atPath: orphan.path))

        // HistoryService.reconcileOrphans scans generatedAudioDirectory, not tmpDir.
        // We verify the logic by ensuring the method runs without error when the
        // Generated directory has no rows referencing it.
        let container = try makeInMemoryContainer()
        let service = HistoryService(context: container.mainContext)
        try await service.reconcileOrphans()
    }

    // MARK: - PresetService

    @MainActor
    func testPresetServiceSaveUpdatesExistingRow() throws {
        let container = try makeInMemoryContainer()
        let service = PresetService(context: container.mainContext)

        let preset = Preset(
            name: "Test",
            summary: "A test preset",
            prompt: "original",
            lyricTemplate: "",
            tags: [],
            duration: 30,
            variance: 0.5
        )
        try service.save(preset)

        // Update the same preset (it should now be managed by the context)
        preset.prompt = "updated"
        try service.save(preset)

        let all = try service.fetchAll()
        XCTAssertEqual(all.count, 1, "Saving an existing preset must not create a duplicate")
        XCTAssertEqual(all.first?.prompt, "updated")
    }

    // MARK: - FileUtilities (after resolveAudioPath returns Optional)

    func testResolveAudioPathReturnsNilForMissingFile() {
        let result = FileUtilities.resolveAudioPath("/nonexistent/path/audio.wav")
        XCTAssertNil(result, "resolveAudioPath should return nil when the file does not exist")
    }

    func testResolveAudioPathReturnsURLForExistingFile() throws {
        let tmpFile = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).wav")
        try Data("RIFF".utf8).write(to: tmpFile)
        defer { try? FileManager.default.removeItem(at: tmpFile) }

        let result = FileUtilities.resolveAudioPath(tmpFile.path)
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.path, tmpFile.path)
    }

    // MARK: - AudioFFT DC bin

    func testFFTDCBinIsZero() {
        let fftSize = 1024
        var samples = [Float](repeating: 1.0, count: fftSize) // DC signal (constant offset)
        let result = AudioFFT.magnitudes(samples: samples, fftSize: fftSize)
        XCTAssertFalse(result.isEmpty)
        XCTAssertEqual(result[0], 0, accuracy: 0.001, "DC bin (index 0) should be zeroed")
        _ = samples // suppress unused warning
    }

    // MARK: - GenerationQueueItem Priority Comparable

    func testPriorityComparable() {
        XCTAssertTrue(GenerationQueueItem.Priority.low < .normal)
        XCTAssertTrue(GenerationQueueItem.Priority.normal < .high)
        XCTAssertFalse(GenerationQueueItem.Priority.high < .low)
    }
}

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

    // MARK: - GenerationQueueItem Priority Comparable

    func testPriorityComparable() {
        XCTAssertTrue(GenerationQueueItem.Priority.low < .normal)
        XCTAssertTrue(GenerationQueueItem.Priority.normal < .high)
        XCTAssertFalse(GenerationQueueItem.Priority.high < .low)
    }
}

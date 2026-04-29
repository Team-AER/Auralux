import AVFoundation
import XCTest
@testable import Auralux

@MainActor
final class AudioPlayerServiceTests: XCTestCase {

    private var service: AudioPlayerService!
    private var fixture: URL!

    override func setUp() async throws {
        try await super.setUp()
        service = AudioPlayerService()
        fixture = try AudioFixtures.sineWave(duration: 1.0)
    }

    override func tearDown() async throws {
        service.stop()
        try? FileManager.default.removeItem(at: fixture)
        service = nil
        fixture = nil
        try await super.tearDown()
    }

    // MARK: - Load

    func testLoadSetsMetadata() throws {
        try service.load(url: fixture)
        XCTAssertGreaterThan(service.duration, 0, "Duration should be populated after load")
        XCTAssertEqual(service.currentTime, 0)
        XCTAssertFalse(service.isPlaying)
    }

    func testLoadMissingFileThrows() {
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("nonexistent-\(UUID().uuidString).wav")
        XCTAssertThrowsError(try service.load(url: missing))
    }

    func testLoadReplacesCurrentTrack() throws {
        try service.load(url: fixture)
        let first = service.duration

        let second = try AudioFixtures.sineWave(duration: 0.3)
        defer { try? FileManager.default.removeItem(at: second) }
        try service.load(url: second)

        XCTAssertNotEqual(service.duration, first, "Loading a shorter file should change duration")
        XCTAssertEqual(service.currentTime, 0, "currentTime should reset on new load")
    }

    // MARK: - Play / Pause / Stop

    func testPlayStartsPlayback() throws {
        try service.load(url: fixture)
        service.play()
        XCTAssertTrue(service.isPlaying)
    }

    func testPauseSuspendsPlayback() throws {
        try service.load(url: fixture)
        service.play()
        service.pause()
        XCTAssertFalse(service.isPlaying)
    }

    func testStopResetsState() throws {
        try service.load(url: fixture)
        service.play()
        service.stop()
        XCTAssertFalse(service.isPlaying)
        XCTAssertEqual(service.currentTime, 0)
    }

    func testPlayIsIdempotent() throws {
        try service.load(url: fixture)
        service.play()
        service.play()  // second call should be a no-op
        XCTAssertTrue(service.isPlaying)
    }

    // MARK: - Seek

    func testSeekUpdatesCurrentTime() throws {
        try service.load(url: fixture)
        service.seek(to: 0.5)
        XCTAssertEqual(service.currentTime, 0.5, accuracy: 0.05)
    }

    func testSeekBeyondEndClamps() throws {
        try service.load(url: fixture)
        service.seek(to: service.duration + 10)
        XCTAssertEqual(service.currentTime, service.duration, accuracy: 0.05)
        XCTAssertFalse(service.isPlaying, "Seeking past end should stop playback")
    }

    func testSeekBeforeStartClamps() throws {
        try service.load(url: fixture)
        service.seek(to: -5)
        XCTAssertEqual(service.currentTime, 0, accuracy: 0.05)
    }

    // MARK: - Shutdown

    func testShutdownStopsEngine() throws {
        try service.load(url: fixture)
        service.play()
        service.shutdown()
        XCTAssertFalse(service.isPlaying)
    }
}

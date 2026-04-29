import XCTest
@testable import Auralux

@MainActor
final class PlaybackDiagnosticsTests: XCTestCase {
    func testStallDetectionWritesSnapshot() throws {
        let diagnosticsDir = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: diagnosticsDir) }
        let diagnostics = PlaybackDiagnosticsService(
            directoryProvider: { diagnosticsDir },
            logSink: { _, _ in }
        )

        let t0 = Date(timeIntervalSince1970: 1_700_000_000)
        diagnostics.startSession(trackPath: "/tmp/song.wav", now: t0)
        diagnostics.monitorProgress(
            currentTime: 0.12,
            duration: 30,
            isPlaying: true,
            engineRunning: true,
            nodePlaying: true,
            now: t0
        )

        diagnostics.monitorProgress(
            currentTime: 0.12,
            duration: 30,
            isPlaying: true,
            engineRunning: true,
            nodePlaying: true,
            now: t0.addingTimeInterval(2.1)
        )

        XCTAssertTrue(diagnostics.events.contains(where: { $0.name == "playback_stall_detected" }))
        XCTAssertNotNil(diagnostics.lastSnapshotURL)
        XCTAssertTrue(FileManager.default.fileExists(atPath: diagnostics.lastSnapshotURL?.path ?? ""))
    }

    func testResumeAfterStallIsRecorded() {
        let diagnosticsDir = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: diagnosticsDir) }
        let diagnostics = PlaybackDiagnosticsService(
            directoryProvider: { diagnosticsDir },
            logSink: { _, _ in }
        )
        let t0 = Date(timeIntervalSince1970: 1_700_000_000)
        diagnostics.startSession(trackPath: "/tmp/song.wav", now: t0)

        diagnostics.monitorProgress(
            currentTime: 0.50,
            duration: 30,
            isPlaying: true,
            engineRunning: true,
            nodePlaying: true,
            now: t0
        )
        diagnostics.monitorProgress(
            currentTime: 0.50,
            duration: 30,
            isPlaying: true,
            engineRunning: true,
            nodePlaying: true,
            now: t0.addingTimeInterval(2.0)
        )
        diagnostics.monitorProgress(
            currentTime: 0.85,
            duration: 30,
            isPlaying: true,
            engineRunning: true,
            nodePlaying: true,
            now: t0.addingTimeInterval(2.2)
        )

        XCTAssertTrue(diagnostics.events.contains(where: { $0.name == "playback_stall_detected" }))
        XCTAssertTrue(diagnostics.events.contains(where: { $0.name == "playback_resumed" }))
    }

    func testManualSnapshotContainsReasonAndTrackPath() throws {
        let diagnosticsDir = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: diagnosticsDir) }
        let diagnostics = PlaybackDiagnosticsService(
            directoryProvider: { diagnosticsDir },
            logSink: { _, _ in }
        )

        let now = Date(timeIntervalSince1970: 1_700_000_000)
        diagnostics.startSession(trackPath: "/tmp/focus.wav", now: now)
        diagnostics.logInfo("play_requested", fields: ["engine_running": "true"], now: now)

        let snapshotURL = diagnostics.persistSnapshot(reason: "manual_test_case", now: now.addingTimeInterval(1))
        XCTAssertNotNil(snapshotURL)

        guard let snapshotURL else {
            XCTFail("Expected snapshot URL")
            return
        }

        let data = try Data(contentsOf: snapshotURL)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let snapshot = try decoder.decode(PlaybackDiagnosticsSnapshot.self, from: data)

        XCTAssertEqual(snapshot.reason, "manual_test_case")
        XCTAssertEqual(snapshot.trackPath, "/tmp/focus.wav")
        XCTAssertGreaterThan(snapshot.events.count, 0)
    }

    private func makeTempDirectory() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("AuraluxPlaybackDiagTests")
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}

import Foundation

enum PlaybackEventLevel: String, Codable, Sendable {
    case info
    case warning
    case error
}

struct PlaybackDiagnosticEvent: Codable, Sendable {
    let timestamp: Date
    let level: PlaybackEventLevel
    let name: String
    let fields: [String: String]
}

struct PlaybackDiagnosticsSnapshot: Codable, Sendable {
    let sessionID: UUID
    let createdAt: Date
    let reason: String
    let trackPath: String?
    let eventCount: Int
    let events: [PlaybackDiagnosticEvent]
}

@MainActor
final class PlaybackDiagnosticsService {
    private(set) var events: [PlaybackDiagnosticEvent] = []
    private(set) var sessionID: UUID?
    private(set) var trackPath: String?
    private(set) var lastSnapshotURL: URL?

    var stallThreshold: TimeInterval = 1.5
    var maxEvents = 1200

    private var lastObservedProgressTime: TimeInterval = 0
    private var lastObservedProgressAt: Date?
    private var didReportStall = false

    private let directoryProvider: () -> URL
    private let fileManager: FileManager
    private let logSink: (LogLevel, String) -> Void

    init(
        directoryProvider: @escaping () -> URL = { FileUtilities.diagnosticsDirectory },
        fileManager: FileManager = .default,
        logSink: @escaping (LogLevel, String) -> Void = { level, message in
            switch level {
            case .debug:
                AppLogger.shared.debug(message, category: .player)
            case .info:
                AppLogger.shared.info(message, category: .player)
            case .warning:
                AppLogger.shared.warning(message, category: .player)
            case .error:
                AppLogger.shared.error(message, category: .player)
            }
        }
    ) {
        self.directoryProvider = directoryProvider
        self.fileManager = fileManager
        self.logSink = logSink
    }

    func startSession(trackPath: String?, now: Date = .now) {
        sessionID = UUID()
        self.trackPath = trackPath
        lastSnapshotURL = nil
        events.removeAll(keepingCapacity: true)
        resetStallState(now: now)
        append(.info, name: "session_started", fields: [
            "track_path": trackPath ?? "nil",
            "session_id": sessionID?.uuidString ?? "nil"
        ], timestamp: now)
    }

    func logInfo(_ name: String, fields: [String: String] = [:], now: Date = .now) {
        append(.info, name: name, fields: fields, timestamp: now)
    }

    func logWarning(_ name: String, fields: [String: String] = [:], now: Date = .now) {
        append(.warning, name: name, fields: fields, timestamp: now)
    }

    func logError(_ name: String, fields: [String: String] = [:], now: Date = .now) {
        append(.error, name: name, fields: fields, timestamp: now)
    }

    func monitorProgress(
        currentTime: TimeInterval,
        duration: TimeInterval,
        isPlaying: Bool,
        engineRunning: Bool,
        nodePlaying: Bool,
        now: Date = .now
    ) {
        guard isPlaying else {
            resetStallState(now: now)
            return
        }

        if lastObservedProgressAt == nil {
            lastObservedProgressTime = currentTime
            lastObservedProgressAt = now
            return
        }

        let delta = currentTime - lastObservedProgressTime
        if delta > 0.01 {
            if didReportStall {
                append(.info, name: "playback_resumed", fields: [
                    "current_time": format(currentTime),
                    "duration": format(duration),
                    "engine_running": "\(engineRunning)",
                    "node_playing": "\(nodePlaying)"
                ], timestamp: now)
            }
            didReportStall = false
            lastObservedProgressTime = currentTime
            lastObservedProgressAt = now
            return
        }

        guard let lastObservedProgressAt else { return }
        let stalledFor = now.timeIntervalSince(lastObservedProgressAt)
        guard stalledFor >= stallThreshold, !didReportStall else { return }

        didReportStall = true
        append(.warning, name: "playback_stall_detected", fields: [
            "stalled_for_s": format(stalledFor),
            "current_time": format(currentTime),
            "duration": format(duration),
            "engine_running": "\(engineRunning)",
            "node_playing": "\(nodePlaying)"
        ], timestamp: now)
        _ = persistSnapshot(reason: "playback_stall_detected", now: now)
    }

    @discardableResult
    func persistSnapshot(reason: String, now: Date = .now) -> URL? {
        let sessionID = self.sessionID ?? UUID()
        if self.sessionID == nil {
            self.sessionID = sessionID
        }

        let snapshot = PlaybackDiagnosticsSnapshot(
            sessionID: sessionID,
            createdAt: now,
            reason: reason,
            trackPath: trackPath,
            eventCount: events.count,
            events: Array(events.suffix(300))
        )

        let directory = directoryProvider()
        let sanitizedReason = reason
            .lowercased()
            .replacingOccurrences(of: " ", with: "_")
            .replacingOccurrences(of: "/", with: "_")
        let fileName = "playback-\(sessionID.uuidString)-\(Int(now.timeIntervalSince1970))-\(sanitizedReason).json"
        let url = directory.appendingPathComponent(fileName)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        guard let data = try? encoder.encode(snapshot) else {
            append(.error, name: "snapshot_write_failed", fields: [
                "reason": reason,
                "error": "encoding failed"
            ], timestamp: now)
            return nil
        }

        // Write synchronously — snapshots are only triggered on stall/error, so the
        // engine is already in a degraded state. The write is a few KB of JSON and
        // completes in microseconds. A background Task would set lastSnapshotURL
        // before the file exists, causing a race in any caller that reads it immediately.
        if !fileManager.fileExists(atPath: directory.path) {
            try? fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        }
        guard (try? data.write(to: url, options: .atomic)) != nil else {
            append(.error, name: "snapshot_write_failed", fields: [
                "reason": reason,
                "error": "disk write failed"
            ], timestamp: now)
            return nil
        }

        lastSnapshotURL = url
        append(.info, name: "snapshot_written", fields: [
            "reason": reason,
            "path": url.path
        ], timestamp: now)

        return url
    }

    private func resetStallState(now: Date) {
        didReportStall = false
        lastObservedProgressTime = 0
        // Reset to nil so the next monitorProgress call properly initializes
        // the stall timer from when playback actually resumes, not from now.
        lastObservedProgressAt = nil
    }

    private func append(
        _ level: PlaybackEventLevel,
        name: String,
        fields: [String: String],
        timestamp: Date
    ) {
        let event = PlaybackDiagnosticEvent(
            timestamp: timestamp,
            level: level,
            name: name,
            fields: fields
        )
        events.append(event)
        if events.count > maxEvents {
            events.removeFirst(events.count - maxEvents)
        }

        let formattedFields = fields
            .sorted { $0.key < $1.key }
            .map { "\($0.key)=\($0.value)" }
            .joined(separator: " ")
        let message = formattedFields.isEmpty
            ? "[Playback] \(name)"
            : "[Playback] \(name) \(formattedFields)"

        switch level {
        case .info:
            logSink(.info, message)
        case .warning:
            logSink(.warning, message)
        case .error:
            logSink(.error, message)
        }
    }

    private func format(_ value: TimeInterval) -> String {
        String(format: "%.3f", value)
    }
}

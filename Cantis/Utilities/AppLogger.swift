import Foundation
import Observation
import os.log

/// Severity levels for log entries.
enum LogLevel: String, CaseIterable, Sendable, Comparable {
    case debug = "DEBUG"
    case info = "INFO"
    case warning = "WARN"
    case error = "ERROR"

    static func < (lhs: LogLevel, rhs: LogLevel) -> Bool {
        let order: [LogLevel] = [.debug, .info, .warning, .error]
        return (order.firstIndex(of: lhs) ?? 0) < (order.firstIndex(of: rhs) ?? 0)
    }
}

/// Categories to group and filter log entries.
enum LogCategory: String, CaseIterable, Sendable {
    case engine = "Engine"
    case inference = "Inference"
    case generation = "Generation"
    case player = "Player"
    case network = "Network"
    case app = "App"
}

/// A single log entry.
struct LogEntry: Identifiable, Sendable {
    let id: UUID
    let timestamp: Date
    let level: LogLevel
    let category: LogCategory
    let message: String

    init(level: LogLevel, category: LogCategory, message: String) {
        self.id = UUID()
        self.timestamp = Date()
        self.level = level
        self.category = category
        self.message = message
    }

    var formattedTimestamp: String {
        Self.formatter.string(from: timestamp)
    }

    private static let formatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss.SSS"
        return f
    }()
}

/// Centralized, observable logger that captures structured log entries.
///
/// All writes funnel through the main actor so SwiftUI views can bind
/// directly to `entries`.  The logger also mirrors every entry to the
/// unified OSLog subsystem.
@MainActor
@Observable
final class AppLogger {
    /// Shared singleton so any part of the app can log without
    /// needing an environment injection.
    static let shared = AppLogger()

    /// All log entries, newest last.
    private(set) var entries: [LogEntry] = []

    /// Maximum entries kept in memory.
    var maxEntries = 2000

    /// Minimum level that is actually recorded.
    var minimumLevel: LogLevel = .debug

    private let osLog = OSLog(subsystem: "com.auralux.app", category: "AppLogger")

    private init() {}

    // MARK: - Convenience shortcuts

    func debug(_ message: String, category: LogCategory = .app) {
        log(level: .debug, category: category, message: message)
    }

    func info(_ message: String, category: LogCategory = .app) {
        log(level: .info, category: category, message: message)
    }

    func warning(_ message: String, category: LogCategory = .app) {
        log(level: .warning, category: category, message: message)
    }

    func error(_ message: String, category: LogCategory = .app) {
        log(level: .error, category: category, message: message)
    }

    // MARK: - Core

    func log(level: LogLevel, category: LogCategory, message: String) {
        guard level >= minimumLevel else { return }

        let entry = LogEntry(level: level, category: category, message: message)
        entries.append(entry)

        // Trim from the front when we exceed the limit.
        if entries.count > maxEntries {
            entries.removeFirst(entries.count - maxEntries)
        }

        // Mirror to system log.
        let osType: OSLogType = switch level {
        case .debug:   .debug
        case .info:    .info
        case .warning: .default
        case .error:   .error
        }
        os_log("%{public}@ [%{public}@] %{public}@", log: osLog, type: osType,
               level.rawValue, category.rawValue, message)
    }

    func clear() {
        entries.removeAll()
    }

    /// Export all current entries as a plain-text string (useful for copy/paste).
    func exportText() -> String {
        entries.map { entry in
            "\(entry.formattedTimestamp) [\(entry.level.rawValue)] [\(entry.category.rawValue)] \(entry.message)"
        }.joined(separator: "\n")
    }
}

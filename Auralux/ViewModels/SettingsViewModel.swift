import Foundation
import Observation

@MainActor
@Observable
final class SettingsViewModel {
    enum QuantizationMode: String, CaseIterable, Identifiable {
        case fp16
        case int8

        var id: String { rawValue }
    }

    var quantizationMode: QuantizationMode = .fp16 {
        didSet { save(key: Keys.quantizationMode, value: quantizationMode.rawValue) }
    }

    var lowMemoryMode = false {
        didSet { save(key: Keys.lowMemoryMode, value: lowMemoryMode) }
    }

    var autoStartServer = true {
        didSet { save(key: Keys.autoStartServer, value: autoStartServer) }
    }

    var maxConcurrentJobs = 1 {
        didSet {
            let clamped = max(1, min(4, maxConcurrentJobs))
            if maxConcurrentJobs != clamped {
                maxConcurrentJobs = clamped  // triggers didSet once more with clamped value
                return
            }
            save(key: Keys.maxConcurrentJobs, value: clamped)
        }
    }

    var defaultExportFormat: AudioExportFormat = .wav {
        didSet { save(key: Keys.defaultExportFormat, value: defaultExportFormat.rawValue) }
    }

    private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        loadAll()
    }

    func resetToDefaults() {
        quantizationMode = .fp16
        lowMemoryMode = false
        autoStartServer = true
        maxConcurrentJobs = 1
        defaultExportFormat = .wav
    }

    // MARK: - Persistence

    private enum Keys {
        static let quantizationMode = "settings.quantizationMode"
        static let lowMemoryMode = "settings.lowMemoryMode"
        static let autoStartServer = "settings.autoStartServer"
        static let maxConcurrentJobs = "settings.maxConcurrentJobs"
        static let defaultExportFormat = "settings.defaultExportFormat"
    }

    private func loadAll() {
        if let raw = defaults.string(forKey: Keys.quantizationMode),
           let mode = QuantizationMode(rawValue: raw) {
            quantizationMode = mode
        }
        if defaults.object(forKey: Keys.lowMemoryMode) != nil {
            lowMemoryMode = defaults.bool(forKey: Keys.lowMemoryMode)
        }
        if defaults.object(forKey: Keys.autoStartServer) != nil {
            autoStartServer = defaults.bool(forKey: Keys.autoStartServer)
        }
        if defaults.object(forKey: Keys.maxConcurrentJobs) != nil {
            let stored = defaults.integer(forKey: Keys.maxConcurrentJobs)
            maxConcurrentJobs = max(1, min(4, stored))
        }
        if let raw = defaults.string(forKey: Keys.defaultExportFormat),
           let format = AudioExportFormat(rawValue: raw) {
            defaultExportFormat = format
        }
    }

    private func save(key: String, value: Any) {
        defaults.set(value, forKey: key)
    }
}

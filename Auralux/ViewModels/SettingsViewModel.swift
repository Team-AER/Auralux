import Foundation
import Observation

@MainActor
@Observable
final class SettingsViewModel {
    enum QuantizationMode: String, CaseIterable, Identifiable {
        case fp16

        var id: String { rawValue }
    }

    var quantizationMode: QuantizationMode = .fp16 {
        didSet { save(key: Keys.quantizationMode, value: quantizationMode.rawValue) }
    }

    var lowMemoryMode = false {
        didSet { save(key: Keys.lowMemoryMode, value: lowMemoryMode) }
    }

    /// Loads the 5Hz audio-token LM (`acestep-5Hz-lm-0.6B`) at app start.
    ///
    /// Off by default: the LM costs ~1.2 GB resident and is currently used by
    /// no generation path. Wiring (FSQ codebooks → detokenize → src_latents)
    /// lands in a follow-up. Toggling this requires a model reload.
    var useLM = false {
        didSet { save(key: Keys.useLM, value: useLM) }
    }

    // MARK: - Generation defaults
    //
    // These seed a fresh `GenerationParameters` when the user opens a new
    // request. They are *not* hard limits — the generation panel can override
    // them per-request.

    var ditVariant: DiTVariant = .turbo {
        didSet {
            save(key: Keys.ditVariant, value: ditVariant.rawValue)
            defaultNumSteps = ditVariant.defaultNumSteps
            defaultCfgScale = ditVariant.defaultCfgScale
        }
    }

    /// When non-nil, the engine loads weights from a user-added custom model
    /// instead of the built-in `ditVariant` directory. The custom model still
    /// inherits `ditVariant`'s config/step/CFG defaults (its base variant).
    var activeCustomModelID: String? = nil {
        didSet {
            if let id = activeCustomModelID {
                save(key: Keys.activeCustomModelID, value: id)
            } else {
                defaults.removeObject(forKey: Keys.activeCustomModelID)
            }
        }
    }

    var defaultMode: GenerationMode = .text2music {
        didSet { save(key: Keys.defaultMode, value: defaultMode.rawValue) }
    }

    var defaultNumSteps: Int = 8 {
        didSet {
            let clamped = max(1, min(ditVariant.maxNumSteps, defaultNumSteps))
            if defaultNumSteps != clamped {
                defaultNumSteps = clamped
                return
            }
            save(key: Keys.defaultNumSteps, value: clamped)
        }
    }

    /// Valid shifts are {1.0, 2.0, 3.0} per upstream `SHIFT_TIMESTEPS`.
    var defaultScheduleShift: Double = 1.0 {
        didSet { save(key: Keys.defaultScheduleShift, value: defaultScheduleShift) }
    }

    /// Default cfgScale; only respected on non-turbo variants. Turbo logs
    /// and ignores anything > 1.0 because CFG is distilled into the weights.
    var defaultCfgScale: Double = 1.0 {
        didSet { save(key: Keys.defaultCfgScale, value: defaultCfgScale) }
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
        lowMemoryMode = AppConstants.isLowMemoryMachine
        useLM = false
        defaultExportFormat = .wav
        ditVariant = .turbo
        activeCustomModelID = nil
        defaultMode = .text2music
        defaultNumSteps = ditVariant.defaultNumSteps
        defaultScheduleShift = 1.0
        defaultCfgScale = ditVariant.defaultCfgScale
    }

    // MARK: - Persistence

    enum Keys {
        static let quantizationMode = "settings.quantizationMode"
        static let lowMemoryMode = "settings.lowMemoryMode"
        static let useLM = "settings.useLM"
        static let defaultExportFormat = "settings.defaultExportFormat"
        static let ditVariant = "settings.ditVariant"
        static let activeCustomModelID = "settings.activeCustomModelID"
        static let defaultMode = "settings.defaultMode"
        static let defaultNumSteps = "settings.defaultNumSteps"
        static let defaultScheduleShift = "settings.defaultScheduleShift"
        static let defaultCfgScale = "settings.defaultCfgScale"
    }

    private func loadAll() {
        if let raw = defaults.string(forKey: Keys.quantizationMode),
           let mode = QuantizationMode(rawValue: raw) {
            quantizationMode = mode
        }
        if defaults.object(forKey: Keys.lowMemoryMode) != nil {
            lowMemoryMode = defaults.bool(forKey: Keys.lowMemoryMode)
        } else if AppConstants.isLowMemoryMachine {
            // First launch on a ≤ 16 GiB Mac — opt them in by default. Setting
            // the property triggers `didSet`, which persists the value, so
            // `AuraluxApp.init()` will read it on the next launch as well.
            lowMemoryMode = true
        }
        if defaults.object(forKey: Keys.useLM) != nil {
            useLM = defaults.bool(forKey: Keys.useLM)
        }
        if let raw = defaults.string(forKey: Keys.ditVariant),
           let variant = DiTVariant(rawValue: raw) {
            ditVariant = variant
        }
        if let id = defaults.string(forKey: Keys.activeCustomModelID), !id.isEmpty {
            activeCustomModelID = id
        }
        if let raw = defaults.string(forKey: Keys.defaultMode),
           let mode = GenerationMode(rawValue: raw) {
            defaultMode = mode
        }
        if defaults.object(forKey: Keys.defaultNumSteps) != nil {
            let stored = defaults.integer(forKey: Keys.defaultNumSteps)
            defaultNumSteps = max(1, min(ditVariant.maxNumSteps, stored))
        }
        if defaults.object(forKey: Keys.defaultScheduleShift) != nil {
            defaultScheduleShift = defaults.double(forKey: Keys.defaultScheduleShift)
        }
        if defaults.object(forKey: Keys.defaultCfgScale) != nil {
            defaultCfgScale = defaults.double(forKey: Keys.defaultCfgScale)
        }
        if let raw = defaults.string(forKey: Keys.defaultExportFormat),
           let format = AudioExportFormat(rawValue: raw),
           format.isAvailable {
            defaultExportFormat = format
        }
    }

    private func save(key: String, value: Any) {
        defaults.set(value, forKey: key)
    }
}

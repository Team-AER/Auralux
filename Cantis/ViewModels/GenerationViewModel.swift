import Foundation
import Observation
import SwiftData

enum GenerationState: Equatable {
    case idle
    case preparing
    case generating
    case completed
    case failed(String)

    var isBusy: Bool {
        switch self {
        case .preparing, .generating:
            return true
        case .idle, .completed, .failed:
            return false
        }
    }
}

@MainActor
@Observable
final class GenerationViewModel {
    var prompt = GenerationParameters.default.prompt
    var lyrics = GenerationParameters.default.lyrics
    var tags: [String] = GenerationParameters.default.tags
    var duration: Double = GenerationParameters.default.duration
    var variance: Double = GenerationParameters.default.variance
    var seedText = ""

    // ── DiT knobs (mirror GenerationParameters) ─────────────────────────────
    var mode: GenerationMode = .text2music
    var numSteps: Int = 8
    var scheduleShift: Double = 1.0
    var cfgScale: Double = 1.0
    var referAudioURL: URL?
    var sourceAudioURL: URL?
    /// Time-ranges (seconds) to repaint. Frames inside any range are
    /// regenerated; everything else is kept from the source audio.
    var repaintRanges: [RepaintRange] = []

    var state: GenerationState = .idle
    var progress: Double = 0
    var progressMessage: String = ""
    var currentJobID: String? = nil
    var lastTrack: GeneratedTrack?

    private let engine: NativeInferenceEngine
    private let log = AppLogger.shared
    private var generationTask: Task<Void, Never>?

    init(engine: NativeInferenceEngine) {
        self.engine = engine
    }

    /// Seed the per-request controls from saved settings. Called by the view
    /// when it appears so users see their configured defaults; per-request
    /// edits stay in this VM and don't leak back into settings.
    func applyDefaults(from settings: SettingsViewModel) {
        mode = settings.defaultMode
        numSteps = settings.defaultNumSteps
        scheduleShift = settings.defaultScheduleShift
        cfgScale = settings.ditVariant.respectsCFG ? settings.defaultCfgScale : 1.0
    }

    /// Restore the per-request sliders to defaults derived from the active
    /// DiT variant and the currently-selected mode. Keeps the user's
    /// mode/prompt/lyrics — only the numeric knobs reset.
    func resetParameters(using settings: SettingsViewModel) {
        duration = GenerationParameters.default.duration
        variance = GenerationParameters.default.variance
        numSteps = settings.defaultNumSteps
        scheduleShift = settings.defaultScheduleShift
        cfgScale = settings.ditVariant.respectsCFG ? settings.defaultCfgScale : 1.0
        seedText = ""
    }

    func applyPreset(_ preset: Preset) {
        prompt = preset.prompt
        lyrics = preset.lyricTemplate
        tags = preset.tags
        duration = preset.duration
        variance = preset.variance
    }

    func addTag(_ tag: String) {
        let normalized = tag.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !normalized.isEmpty else { return }
        guard !tags.contains(normalized) else { return }
        tags.append(normalized)
    }

    func removeTag(_ tag: String) {
        tags.removeAll { $0 == tag }
    }

    func generate(in context: ModelContext) {
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            state = .failed("Prompt is required.")
            return
        }

        generationTask?.cancel()
        state = .preparing
        progress = 0
        progressMessage = "Preparing..."
        currentJobID = nil

        // Resolve the seed up front so the user can see the actual value that
        // was used (the "Random" placeholder is replaced by a concrete number).
        if Int(seedText) == nil {
            seedText = String(Self.randomSeedValue())
        }

        let request = GenerationParameters(
            prompt: prompt,
            lyrics: lyrics,
            tags: tags,
            duration: duration,
            variance: variance,
            seed: Int(seedText),
            mode: mode,
            numSteps: numSteps,
            scheduleShift: scheduleShift,
            cfgScale: cfgScale,
            referAudioURL: referAudioURL,
            sourceAudioURL: sourceAudioURL,
            repaintMaskRanges: repaintRanges
        )

        log.info("Starting generation: \"\(prompt.prefix(60))\" duration=\(duration)s", category: .generation)

        let stream = engine.generate(request: request)

        generationTask = Task {
            defer { generationTask = nil }
            do {
                let historyService = HistoryService(context: context)
                for try await event in stream {
                    switch event {
                    case .preparing(let message):
                        state = .preparing
                        progressMessage = message
                    case .step(let current, let total):
                        state = .generating
                        progress = Double(current) / Double(total)
                        progressMessage = "Step \(current) / \(total)"
                    case .saving:
                        progressMessage = "Saving audio..."
                    case .completed(let audioURL):
                        let track = GeneratedTrack(
                            title: request.prompt,
                            prompt: request.prompt,
                            lyrics: request.lyrics,
                            tags: request.tags,
                            duration: request.duration,
                            variance: request.variance,
                            seed: request.seed,
                            generationID: UUID().uuidString,
                            audioFilePath: FileUtilities.relativeAudioPath(from: audioURL.path)
                        )
                        try historyService.insert(track)
                        await Task.yield()
                        lastTrack = track
                    }
                }
                state = .completed
                log.info("Generation completed successfully", category: .generation)
            } catch is CancellationError {
                state = .idle
                log.info("Generation cancelled", category: .generation)
            } catch {
                state = .failed(error.localizedDescription)
                log.error("Generation failed: \(error.localizedDescription)", category: .generation)
            }
        }
    }

    /// Replace the seed with a freshly-randomized value so the user can keep
    /// rolling without having to clear the field manually.
    func randomizeSeed() {
        seedText = String(Self.randomSeedValue())
    }

    private static func randomSeedValue() -> UInt32 {
        UInt32.random(in: 0...UInt32.max)
    }

    func cancel() {
        generationTask?.cancel()
        engine.cancelGeneration()
        log.info("Generation cancelled by user", category: .generation)
        state = .idle
        progress = 0
        progressMessage = ""
        currentJobID = nil
    }
}

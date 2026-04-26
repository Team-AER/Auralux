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

    var state: GenerationState = .idle
    var progress: Double = 0
    var progressMessage: String = ""
    var currentJobID: String?
    var lastTrack: GeneratedTrack?

    private let inferenceService: InferenceService
    private let log = AppLogger.shared
    private var generationTask: Task<Void, Never>?

    init(inferenceService: InferenceService) {
        self.inferenceService = inferenceService
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

    func generate(in context: ModelContext, engine: EngineService) {
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            state = .failed("Prompt is required.")
            return
        }

        generationTask?.cancel()
        state = .preparing
        progress = 0
        progressMessage = "Submitting..."

        let request = GenerationRequest(
            prompt: prompt,
            lyrics: lyrics,
            tags: tags,
            duration: duration,
            variance: variance,
            seed: Int(seedText)
        )

        log.info("Starting generation: \"\(prompt.prefix(60))\" duration=\(duration)s", category: .generation)

        generationTask = Task {
            do {
                progressMessage = "Starting inference engine..."
                try await engine.prepareForGeneration()

                let submission = try await inferenceService.generate(request)
                currentJobID = submission.jobID
                state = .generating
                log.info("Job submitted: \(submission.jobID)", category: .generation)
                try await pollUntilComplete(jobID: submission.jobID, request: request, context: context)
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

    func cancel() {
        generationTask?.cancel()
        if let currentJobID {
            let jobID = currentJobID
            Task {
                do {
                    try await inferenceService.cancel(jobID: jobID)
                } catch {
                    log.warning("Server cancel request failed for job \(jobID): \(error.localizedDescription)", category: .generation)
                }
            }
        }
        log.info("Generation cancelled by user", category: .generation)
        state = .idle
        progress = 0
        progressMessage = ""
        currentJobID = nil
    }

    private func pollUntilComplete(jobID: String, request: GenerationRequest, context: ModelContext) async throws {
        let historyService = HistoryService(context: context)
        var consecutiveErrors = 0
        let maxConsecutiveErrors = 10
        var lastLoggedStatus = ""
        var pollCount = 0

        while !Task.isCancelled {
            do {
                let status = try await inferenceService.poll(jobID: jobID)
                consecutiveErrors = 0
                pollCount += 1
                progress = status.progress
                progressMessage = status.message ?? ""

                // Log every poll so the user can see what's happening
                let statusSummary = "[\(status.status)] \(Int(status.progress * 100))% — \(status.message ?? "")"
                if statusSummary != lastLoggedStatus {
                    log.info("Poll #\(pollCount): \(statusSummary)", category: .generation)
                    lastLoggedStatus = statusSummary
                } else if pollCount % 10 == 0 {
                    // Even if unchanged, log periodically so it's clear we're still alive
                    log.debug("Poll #\(pollCount): still \(statusSummary)", category: .generation)
                }

                if status.status == "failed" {
                    throw NSError(domain: "Auralux", code: -1, userInfo: [NSLocalizedDescriptionKey: status.message ?? "Generation failed"])
                }

                if status.status == "cancelled" {
                    throw CancellationError()
                }

                if status.status == "completed" {
                    let storedPath: String? = {
                        guard let raw = status.audioPath else { return nil }
                        return FileUtilities.relativeAudioPath(from: raw)
                    }()
                    let track = GeneratedTrack(
                        title: request.prompt,
                        prompt: request.prompt,
                        lyrics: request.lyrics,
                        tags: request.tags,
                        duration: request.duration,
                        variance: request.variance,
                        seed: request.seed,
                        generationID: jobID,
                        audioFilePath: storedPath
                    )

                    // Persist first, then yield so SwiftData change
                    // notifications settle before we touch Observable
                    // properties that trigger the PlayerView (and its
                    // AVAudioEngine setup).
                    try historyService.insert(track)
                    await Task.yield()

                    lastTrack = track
                    currentJobID = nil
                    return
                }
            } catch let error as NSError where error.domain == "Auralux" {
                throw error
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                // If the server crashed and restarted, the job is gone — fail immediately
                if case InferenceError.jobNotFound = error {
                    log.error("Job lost — server crashed during generation", category: .generation)
                    throw error
                }
                consecutiveErrors += 1
                log.warning("Poll error (\(consecutiveErrors)/\(maxConsecutiveErrors)): \(error.localizedDescription)", category: .network)
                if consecutiveErrors >= maxConsecutiveErrors {
                    log.error("Too many consecutive poll errors, aborting", category: .network)
                    throw InferenceError.requestFailed("Lost connection to server after \(maxConsecutiveErrors) retries")
                }
            }

            // Adaptive poll interval: 2s normally, back off on errors
            let interval = consecutiveErrors > 0
                ? min(Double(consecutiveErrors) * 2.0, 10.0)
                : 2.0
            try await Task.sleep(for: .seconds(interval))
        }

        throw CancellationError()
    }
}

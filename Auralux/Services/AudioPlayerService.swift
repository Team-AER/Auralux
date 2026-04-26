import AVFoundation
import Foundation
import Observation

@MainActor
@Observable
final class AudioPlayerService {
    var isPlaying = false
    var isLooping = false
    var currentTime: TimeInterval = 0
    var duration: TimeInterval = 0
    var volume: Float = 1 {
        didSet {
            if engineReady { engine.mainMixerNode.outputVolume = volume }
        }
    }

    /// Spectrum data for the player UI.
    /// Live tap-based analysis is intentionally disabled because it was blocking playback startup.
    var spectrumMagnitudes: [Float] = []

    private var engine = AVAudioEngine()
    private var playerNode = AVAudioPlayerNode()
    private var audioFile: AVAudioFile?
    private var progressTask: Task<Void, Never>?
    private var engineReady = false
    private var hasScheduledAudio = false
    private var scheduledStartTime: TimeInterval = 0
    private let diagnostics = PlaybackDiagnosticsService()

    init() {
        // Audio engine setup is deferred to first use to avoid
        // GCD queue assertion failures during app startup.
    }

    /// Stops playback and shuts down the audio engine. Call from the app lifecycle
    /// (applicationWillTerminate) — Swift 6 prohibits accessing @MainActor-isolated
    /// properties from deinit.
    func shutdown() {
        stop()  // resets isPlaying, currentTime, cancels progressTask
        if engine.isRunning { engine.stop() }
        engineReady = false
        diagnostics.logInfo("engine_shutdown")
    }

    /// Ensures the audio engine is configured and running.
    /// Called lazily before any audio operation.
    private func ensureEngine() {
        guard !engineReady else { return }
        engine.attach(playerNode)
        engine.connect(playerNode, to: engine.mainMixerNode, format: nil)
        engine.mainMixerNode.outputVolume = volume
        engineReady = true
        diagnostics.logInfo("engine_configured")
    }

    /// Starts the engine only when actually needed for playback.
    private func ensureEngineRunning() {
        ensureEngine()

        guard !engine.isRunning else {
            diagnostics.logInfo("engine_already_running")
            return
        }
        do {
            try engine.start()
            diagnostics.logInfo("engine_started")
        } catch {
            NSLog("Audio engine failed to start: \(error)")
            diagnostics.logError("engine_start_failed", fields: ["error": error.localizedDescription])
            _ = diagnostics.persistSnapshot(reason: "engine_start_failed")
        }
    }

    func load(url: URL) throws {
        diagnostics.startSession(trackPath: url.path)
        diagnostics.logInfo("load_requested", fields: ["path": url.path])
        ensureEngine()
        do {
            let file = try AVAudioFile(forReading: url)
            playerNode.stop()
            isPlaying = false
            progressTask?.cancel()
            progressTask = nil
            audioFile = file
            duration = Double(file.length) / file.fileFormat.sampleRate
            currentTime = 0
            scheduledStartTime = 0
            hasScheduledAudio = false
            spectrumMagnitudes = []
            diagnostics.logInfo("load_succeeded", fields: [
                "duration": String(format: "%.3f", duration),
                "sample_rate": String(format: "%.1f", file.fileFormat.sampleRate),
                "frames": "\(file.length)"
            ])
        } catch {
            diagnostics.logError("load_failed", fields: ["error": error.localizedDescription])
            _ = diagnostics.persistSnapshot(reason: "load_failed")
            throw error
        }
    }

    func play() {
        diagnostics.logInfo("play_requested", fields: [
            "engine_ready": "\(engineReady)",
            "engine_running": "\(engine.isRunning)",
            "node_playing": "\(playerNode.isPlaying)"
        ])
        guard !isPlaying else {
            diagnostics.logInfo("play_ignored_already_playing")
            return
        }
        guard audioFile != nil else {
            diagnostics.logWarning("play_ignored_no_audio_file")
            return
        }
        ensureEngine()
        if !hasScheduledAudio {
            scheduleFile()
        }
        ensureEngineRunning()
        playerNode.play()
        isPlaying = true
        startProgressUpdates()
        diagnostics.logInfo("play_started", fields: [
            "engine_running": "\(engine.isRunning)",
            "node_playing": "\(playerNode.isPlaying)"
        ])
    }

    func pause() {
        playerNode.pause()
        isPlaying = false
        progressTask?.cancel()
        progressTask = nil
        diagnostics.logInfo("playback_paused", fields: ["current_time": String(format: "%.3f", currentTime)])
    }

    func stop() {
        playerNode.stop()
        isPlaying = false
        currentTime = 0
        scheduledStartTime = 0
        hasScheduledAudio = false
        spectrumMagnitudes = []
        progressTask?.cancel()
        progressTask = nil
        diagnostics.logInfo("playback_stopped")
    }

    func seek(to time: TimeInterval) {
        diagnostics.logInfo("seek_requested", fields: ["target_time": String(format: "%.3f", time)])
        ensureEngineRunning()
        guard let audioFile else {
            diagnostics.logWarning("seek_ignored_no_audio_file")
            return
        }
        let wasPlaying = isPlaying
        playerNode.stop()

        let sampleRate = audioFile.processingFormat.sampleRate
        let targetFrame = AVAudioFramePosition(time * sampleRate)
        let totalFrames = AVAudioFramePosition(audioFile.length)
        let clampedFrame = max(0, min(targetFrame, totalFrames))
        let remainingFrames = AVAudioFrameCount(totalFrames - clampedFrame)

        guard remainingFrames > 0 else {
            currentTime = duration
            isPlaying = false
            hasScheduledAudio = false
            diagnostics.logInfo("seek_clamped_to_end")
            return
        }

        scheduledStartTime = Double(clampedFrame) / sampleRate
        playerNode.scheduleSegment(audioFile, startingFrame: clampedFrame, frameCount: remainingFrames, at: nil) { [weak self] in
            guard let self else { return }
            Task { @MainActor in
                let finishedAt = self.currentTime
                self.isPlaying = false
                self.hasScheduledAudio = false
                self.progressTask?.cancel()
                self.progressTask = nil
                self.currentTime = 0
                self.diagnostics.logInfo("segment_finished", fields: [
                    "looping": "\(self.isLooping)",
                    "finished_at": String(format: "%.3f", finishedAt),
                    "duration": String(format: "%.3f", self.duration)
                ])
                if self.isLooping {
                    self.scheduleFile()
                    self.play()
                }
            }
        }

        hasScheduledAudio = true
        currentTime = scheduledStartTime
        if wasPlaying {
            playerNode.play()
            isPlaying = true
            startProgressUpdates()
            diagnostics.logInfo("seek_resumed_playback")
        }
    }

    private func scheduleFile() {
        guard let audioFile else {
            diagnostics.logWarning("schedule_ignored_no_audio_file")
            return
        }
        playerNode.stop()
        scheduledStartTime = 0
        playerNode.scheduleFile(audioFile, at: nil) { [weak self] in
            guard let self else { return }
            Task { @MainActor in
                let finishedAt = self.currentTime
                self.isPlaying = false
                self.hasScheduledAudio = false
                self.progressTask?.cancel()
                self.progressTask = nil
                self.currentTime = 0
                self.diagnostics.logInfo("file_finished", fields: [
                    "looping": "\(self.isLooping)",
                    "finished_at": String(format: "%.3f", finishedAt),
                    "duration": String(format: "%.3f", self.duration)
                ])
                if self.isLooping {
                    self.scheduleFile()
                    self.play()
                }
            }
        }
        hasScheduledAudio = true
        diagnostics.logInfo("file_scheduled", fields: ["duration": String(format: "%.3f", duration)])
    }

    private func updateProgress() {
        guard isPlaying else { return }

        guard let nodeTime = playerNode.lastRenderTime,
              let playerTime = playerNode.playerTime(forNodeTime: nodeTime)
        else {
            diagnostics.monitorProgress(
                currentTime: currentTime,
                duration: duration,
                isPlaying: isPlaying,
                engineRunning: engine.isRunning,
                nodePlaying: playerNode.isPlaying
            )
            return
        }

        let time = scheduledStartTime + Double(playerTime.sampleTime) / playerTime.sampleRate
        currentTime = min(duration, time)
        diagnostics.monitorProgress(
            currentTime: currentTime,
            duration: duration,
            isPlaying: isPlaying,
            engineRunning: engine.isRunning,
            nodePlaying: playerNode.isPlaying
        )
    }

    private func startProgressUpdates() {
        progressTask?.cancel()
        diagnostics.logInfo("progress_updates_started")
        progressTask = Task {
            while !Task.isCancelled {
                updateProgress()
                try? await Task.sleep(for: .milliseconds(50))
            }
        }
    }

    @discardableResult
    func captureDiagnostics(reason: String = "manual_capture") -> URL? {
        diagnostics.persistSnapshot(reason: reason)
    }
}

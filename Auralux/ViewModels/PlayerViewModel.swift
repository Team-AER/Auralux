import AVFoundation
import Foundation
import Observation

@MainActor
@Observable
final class PlayerViewModel {
    let playerService: AudioPlayerService
    var loadedPath: String?
    var errorMessage: String?

    var isPlaying: Bool { playerService.isPlaying }
    var currentTime: TimeInterval { playerService.currentTime }
    var duration: TimeInterval { playerService.duration }
    var isLooping: Bool {
        get { playerService.isLooping }
        set { playerService.isLooping = newValue }
    }

    var progress: Double {
        guard duration > 0 else { return 0 }
        return currentTime / duration
    }

    var waveformSamples: [Float] = []

    init(playerService: AudioPlayerService = AudioPlayerService()) {
        self.playerService = playerService
    }

    func load(path: String?) {
        guard let path else { return }
        errorMessage = nil
        do {
            let url = URL(fileURLWithPath: path)
            try playerService.load(url: url)
            loadedPath = path
            Task {
                waveformSamples = await Self.extractWaveform(from: url, targetSampleCount: 200)
            }
        } catch {
            errorMessage = "Failed to load audio: \(error.localizedDescription)"
        }
    }

    func playPause() {
        errorMessage = nil
        if isPlaying {
            playerService.pause()
        } else {
            playerService.play()
        }
    }

    func stop() {
        playerService.stop()
    }

    func seek(to fraction: Double) {
        let target = max(0, min(1, fraction)) * duration
        playerService.seek(to: target)
    }

    func clearError() {
        errorMessage = nil
    }

    /// Downsamples audio file into an array of amplitude values for waveform rendering.
    static func extractWaveform(from url: URL, targetSampleCount: Int) async -> [Float] {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                guard let file = try? AVAudioFile(forReading: url) else {
                    continuation.resume(returning: [])
                    return
                }
                let totalFrames = AVAudioFrameCount(file.length)
                guard totalFrames > 0,
                      let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: totalFrames)
                else {
                    continuation.resume(returning: [])
                    return
                }
                do {
                    try file.read(into: buffer)
                } catch {
                    continuation.resume(returning: [])
                    return
                }
                guard let channelData = buffer.floatChannelData?[0] else {
                    continuation.resume(returning: [])
                    return
                }
                let frameCount = Int(buffer.frameLength)
                let samplesPerBin = max(1, frameCount / targetSampleCount)
                var result: [Float] = []
                result.reserveCapacity(targetSampleCount)

                for bin in 0..<targetSampleCount {
                    let start = bin * samplesPerBin
                    if start >= frameCount { break }
                    let end = min(frameCount, start + samplesPerBin)
                    var peak: Float = 0
                    for i in start..<end {
                        let sample = abs(channelData[i])
                        if sample > peak { peak = sample }
                    }
                    result.append(peak)
                }
                continuation.resume(returning: result)
            }
        }
    }
}

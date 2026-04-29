import AVFoundation
import Accelerate
import Foundation
import Observation

@MainActor
@Observable
final class PlayerViewModel {
    let playerService: AudioPlayerService
    var loadedPath: String?
    var errorMessage: String?
    private var waveformTask: Task<Void, Never>?
    private let log = AppLogger.shared

    var isPlaying: Bool { playerService.isPlaying }
    var currentTime: TimeInterval { playerService.currentTime }
    var duration: TimeInterval { playerService.duration }
    var isLooping: Bool {
        get { playerService.isLooping }
        set { playerService.isLooping = newValue }
    }
    var volume: Float {
        get { playerService.volume }
        set { playerService.volume = max(0, min(1, newValue)) }
    }

    var progress: Double {
        guard duration > 0 else { return 0 }
        return currentTime / duration
    }

    var waveformSamples: [Float] = []
    private var spectrogram: [[Float]] = []
    private static let spectrogramFPS: Double = 20

    var spectrumBins: [Float] {
        guard isPlaying, !spectrogram.isEmpty, duration > 0 else { return [] }
        let idx = max(0, min(spectrogram.count - 1, Int(currentTime * Self.spectrogramFPS)))
        return spectrogram[idx]
    }

    init(playerService: AudioPlayerService = AudioPlayerService()) {
        self.playerService = playerService
    }

    func load(path: String?) {
        guard let path else { return }
        guard loadedPath != path else { return }
        errorMessage = nil
        waveformTask?.cancel()
        waveformSamples = []
        spectrogram = []
        log.info("PlayerViewModel load path=\(path)", category: .player)
        guard let url = FileUtilities.resolveAudioPath(path) else {
            loadedPath = nil
            errorMessage = "Audio file not found."
            log.warning("PlayerViewModel: audio file missing for path=\(path)", category: .player)
            return
        }
        do {
            try playerService.load(url: url)
            loadedPath = path
            waveformTask = Task {
                async let waveformResult = Self.extractWaveform(from: url, targetSampleCount: 200)
                async let spectrogramResult = Self.extractSpectrogram(from: url, fps: Self.spectrogramFPS)
                let (samples, spectro) = await (waveformResult, spectrogramResult)
                guard !Task.isCancelled else { return }
                waveformSamples = samples
                spectrogram = spectro
                await MainActor.run {
                    self.log.info("Extraction complete waveform=\(samples.count) spectrogram=\(spectro.count)", category: .player)
                }
            }
        } catch {
            loadedPath = nil
            errorMessage = "Failed to load audio: \(error.localizedDescription)"
            log.error("PlayerViewModel load failed: \(error.localizedDescription)", category: .player)
            _ = playerService.captureDiagnostics(reason: "load_failed_ui")
        }
    }

    func playPause() {
        errorMessage = nil
        if isPlaying {
            log.info("PlayerViewModel pause requested", category: .player)
            playerService.pause()
        } else {
            log.info("PlayerViewModel play requested", category: .player)
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

    @discardableResult
    func capturePlaybackDiagnostics(reason: String = "manual_capture_ui") -> URL? {
        let snapshotURL = playerService.captureDiagnostics(reason: reason)
        if let snapshotURL {
            log.warning("Playback diagnostics snapshot saved: \(snapshotURL.path)", category: .player)
        } else {
            log.error("Failed to capture playback diagnostics snapshot", category: .player)
        }
        return snapshotURL
    }

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
                do { try file.read(into: buffer) } catch {
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
                        let s = abs(channelData[i])
                        if s > peak { peak = s }
                    }
                    result.append(peak)
                }
                if let maxVal = result.max(), maxVal > 0 {
                    let scale = 1.0 / maxVal
                    for i in result.indices { result[i] *= scale }
                }
                continuation.resume(returning: result)
            }
        }
    }

    static func extractSpectrogram(from url: URL, fps: Double = 20, binCount: Int = 36) async -> [[Float]] {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .utility).async {
                guard let file = try? AVAudioFile(forReading: url) else {
                    continuation.resume(returning: [])
                    return
                }
                let sampleRate = file.processingFormat.sampleRate
                let framesPerSlice = AVAudioFrameCount(max(1, sampleRate / fps))
                let fftSize = 2048
                let halfSize = fftSize / 2
                let log2n = vDSP_Length(log2(Float(fftSize)))
                guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
                    continuation.resume(returning: [])
                    return
                }
                defer { vDSP_destroy_fftsetup(fftSetup) }

                var hannWindow = [Float](repeating: 0, count: fftSize)
                vDSP_hann_window(&hannWindow, vDSP_Length(fftSize), Int32(vDSP_HANN_NORM))
                var windowed = [Float](repeating: 0, count: fftSize)
                var realPart = [Float](repeating: 0, count: halfSize)
                var imagPart = [Float](repeating: 0, count: halfSize)

                let sliceCount = Int(Double(file.length) / Double(framesPerSlice)) + 1
                var result = [[Float]]()
                result.reserveCapacity(sliceCount)

                let format = file.processingFormat
                while file.framePosition < file.length {
                    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: framesPerSlice),
                          (try? file.read(into: buffer, frameCount: framesPerSlice)) != nil,
                          buffer.frameLength > 0,
                          let ch = buffer.floatChannelData?[0]
                    else { break }

                    let n = min(Int(buffer.frameLength), fftSize)
                    vDSP_vmul(ch, 1, hannWindow, 1, &windowed, 1, vDSP_Length(n))
                    if n < fftSize { for i in n..<fftSize { windowed[i] = 0 } }

                    for i in 0..<halfSize {
                        realPart[i] = windowed[i << 1]
                        imagPart[i] = windowed[(i << 1) | 1]
                    }

                    realPart.withUnsafeMutableBufferPointer { rBuf in
                        imagPart.withUnsafeMutableBufferPointer { iBuf in
                            var split = DSPSplitComplex(realp: rBuf.baseAddress!, imagp: iBuf.baseAddress!)
                            vDSP_fft_zrip(fftSetup, &split, 1, log2n, FFTDirection(FFT_FORWARD))
                        }
                    }

                    let scale = 1.0 / Double(fftSize)
                    var bins = [Float](repeating: 0, count: binCount)
                    for i in 0..<binCount {
                        let lo = Int(pow(Double(i)     / Double(binCount), 2.0) * Double(halfSize))
                        let hi = max(lo + 1, Int(pow(Double(i + 1) / Double(binCount), 2.0) * Double(halfSize)))
                        var peak: Double = 0
                        for k in lo..<min(hi, halfSize) {
                            let m = Double(realPart[k] * realPart[k] + imagPart[k] * imagPart[k]) * scale
                            if m > peak { peak = m }
                        }
                        let dB = peak > 1e-20 ? 10.0 * log10(peak) : -100.0
                        bins[i] = Float(max(0.0, min(1.0, (dB + 80.0) / 80.0)))
                    }
                    result.append(bins)
                }
                continuation.resume(returning: result)
            }
        }
    }
}

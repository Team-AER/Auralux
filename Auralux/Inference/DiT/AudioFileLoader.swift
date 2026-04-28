@preconcurrency import AVFoundation
import MLX
import Foundation

enum AudioFileLoaderError: Error, CustomStringConvertible {
    case openFailed(URL, String)
    case unsupported(URL, String)
    case empty(URL)

    var description: String {
        switch self {
        case .openFailed(let url, let why):
            return "Couldn't open audio file at \(url.lastPathComponent): \(why)"
        case .unsupported(let url, let why):
            return "Unsupported audio format \(url.pathExtension): \(why)"
        case .empty(let url):
            return "Audio file \(url.lastPathComponent) decoded to zero samples"
        }
    }
}

/// Loads any AVFoundation-readable audio file (WAV, MP3, M4A, AAC, FLAC, ...)
/// and returns it as a 48 kHz stereo float32 MLXArray suitable for the
/// `DCHiFiGANEncoder`.
///
/// The AceStep v1.5 turbo VAE expects:
///   * sample rate = 48 000 Hz
///   * 2 channels (stereo)
///   * length a multiple of 1920 samples (one latent frame at 25 Hz)
///
/// Mono inputs are duplicated to stereo. Length is right-padded with silence
/// to satisfy the 1920-multiple constraint.
enum AudioFileLoader {

    /// Target sample rate enforced by the VAE encoder (48 000 Hz).
    static let targetSampleRate: Double = 48_000
    /// One latent frame's worth of waveform samples (48 kHz / 25 Hz).
    static let samplesPerLatentFrame: Int = 1920

    /// Load + convert + pad audio to a `[1, T, 2]` float32 MLXArray.
    ///
    /// - Parameters:
    ///   - url:           Source file (any AVFoundation-readable format).
    ///   - maxDuration:   Optional ceiling on duration in seconds. Trims after
    ///                    decoding so we don't allocate huge buffers for long
    ///                    files we'll only sample a few seconds from.
    /// - Returns: MLXArray of shape `[1, T_audio, 2]` where `T_audio` is a
    ///   multiple of `samplesPerLatentFrame`.
    static func load(
        url: URL,
        maxDuration: TimeInterval? = nil
    ) throws -> MLXArray {
        // Open with AVAudioFile — supports WAV/AIFF/AAC/M4A/MP3/CAF/FLAC.
        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: url)
        } catch {
            throw AudioFileLoaderError.openFailed(url, error.localizedDescription)
        }

        // Convert to 48 kHz, stereo, float32 non-interleaved on the fly.
        let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate:   targetSampleRate,
            channels:     2,
            interleaved:  false
        )
        guard let outputFormat else {
            throw AudioFileLoaderError.unsupported(url, "Couldn't construct 48k stereo float32 format")
        }

        guard let converter = AVAudioConverter(from: file.processingFormat, to: outputFormat) else {
            throw AudioFileLoaderError.unsupported(
                url, "AVAudioConverter rejected \(file.processingFormat) → 48k stereo float32"
            )
        }

        // Read in 1-second chunks at the input rate so memory stays sane.
        let inputRate = file.processingFormat.sampleRate
        let chunkFrames = AVAudioFrameCount(inputRate)

        guard let inputBuf = AVAudioPCMBuffer(
            pcmFormat: file.processingFormat,
            frameCapacity: chunkFrames
        ) else {
            throw AudioFileLoaderError.unsupported(url, "Couldn't allocate input PCM buffer")
        }

        let totalInputFrames = file.length
        if totalInputFrames == 0 {
            throw AudioFileLoaderError.empty(url)
        }

        // Compute conversion ratio + output capacity.
        let ratio = targetSampleRate / inputRate
        let cap = AVAudioFrameCount(Double(chunkFrames) * ratio + 1024)
        guard let outputBuf = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: cap) else {
            throw AudioFileLoaderError.unsupported(url, "Couldn't allocate output PCM buffer")
        }

        // Optional duration clamp at the OUTPUT rate (frames after resample).
        let maxOutputFrames: Int = {
            guard let maxDuration else { return Int.max }
            return Int(maxDuration * targetSampleRate)
        }()

        var leftSamples:  [Float] = []
        var rightSamples: [Float] = []
        leftSamples.reserveCapacity(min(maxOutputFrames, Int(Double(totalInputFrames) * ratio)))
        rightSamples.reserveCapacity(leftSamples.capacity)

        // Box mutable state so the @Sendable converter callback can mutate it
        // without falling foul of Swift 6 concurrency checks.
        final class FeedState: @unchecked Sendable {
            var done = false
        }
        let feed = FeedState()
        let inputBlock: AVAudioConverterInputBlock = { _, status in
            do {
                inputBuf.frameLength = 0
                try file.read(into: inputBuf)
            } catch {
                status.pointee = .endOfStream
                return nil
            }
            if inputBuf.frameLength == 0 {
                status.pointee = .endOfStream
                feed.done = true
                return nil
            }
            status.pointee = .haveData
            return inputBuf
        }

        // Drain the converter until end-of-stream.
        while leftSamples.count < maxOutputFrames {
            outputBuf.frameLength = 0
            var convError: NSError?
            let status = converter.convert(
                to: outputBuf,
                error: &convError,
                withInputFrom: inputBlock
            )
            if let convError {
                throw AudioFileLoaderError.openFailed(url, convError.localizedDescription)
            }
            let n = Int(outputBuf.frameLength)
            if n > 0,
               let chData = outputBuf.floatChannelData {
                let lp = chData[0]
                let rp = chData[1]
                let take = min(n, maxOutputFrames - leftSamples.count)
                leftSamples.append(contentsOf: UnsafeBufferPointer(start: lp, count: take))
                rightSamples.append(contentsOf: UnsafeBufferPointer(start: rp, count: take))
            }
            if status == .endOfStream || (feed.done && n == 0) { break }
            if status == .error { break }
        }

        if leftSamples.isEmpty {
            throw AudioFileLoaderError.empty(url)
        }

        // Right-pad to a multiple of 1920 samples.
        let remainder = leftSamples.count % samplesPerLatentFrame
        if remainder != 0 {
            let pad = samplesPerLatentFrame - remainder
            leftSamples.append(contentsOf: repeatElement(0, count: pad))
            rightSamples.append(contentsOf: repeatElement(0, count: pad))
        }

        // Pack into MLX [1, T, 2] (NLC layout).
        let T = leftSamples.count
        var interleaved = [Float](repeating: 0, count: T * 2)
        for i in 0..<T {
            interleaved[i * 2]     = leftSamples[i]
            interleaved[i * 2 + 1] = rightSamples[i]
        }
        return MLXArray(interleaved, [1, T, 2])
    }
}

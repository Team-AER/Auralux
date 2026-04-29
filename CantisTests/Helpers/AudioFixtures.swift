import AVFoundation
import Foundation

/// Creates real, playable WAV files for use in unit tests.
enum AudioFixtures {

    /// Writes a short mono sine-wave WAV to a temp file and returns its URL.
    /// - Parameters:
    ///   - frequency: Sine frequency in Hz (default 440 Hz = A4).
    ///   - duration: Length in seconds (default 0.5 s — short but non-trivial).
    ///   - sampleRate: Sample rate in Hz (default 44100).
    static func sineWave(
        frequency: Double = 440,
        duration: Double = 0.5,
        sampleRate: Double = 44100
    ) throws -> URL {
        let frameCount = AVAudioFrameCount(duration * sampleRate)
        let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)!
        buffer.frameLength = frameCount

        let data = buffer.floatChannelData![0]
        for i in 0..<Int(frameCount) {
            data[i] = Float(sin(2.0 * .pi * frequency * Double(i) / sampleRate)) * 0.5
        }

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("auralux-test-\(UUID().uuidString).wav")
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        try file.write(from: buffer)
        return url
    }
}

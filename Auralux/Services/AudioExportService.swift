@preconcurrency import AVFoundation
import Foundation

enum AudioExportFormat: String, CaseIterable, Identifiable, Codable {
    case wav
    case flac
    case mp3
    case aac
    case alac

    var id: String { rawValue }

    /// Whether the system can encode this format via AVAssetWriter.
    /// MP3 and FLAC are unsupported — they crash if passed to AVAssetWriter.
    var isAvailable: Bool {
        switch self {
        case .mp3, .flac: return false
        default: return true
        }
    }

    var fileExtension: String {
        switch self {
        case .aac, .alac: return "m4a"
        default: return rawValue
        }
    }

    var audioFormatID: AudioFormatID {
        switch self {
        case .wav: return kAudioFormatLinearPCM
        case .flac: return kAudioFormatFLAC
        case .mp3: return kAudioFormatMPEGLayer3
        case .aac: return kAudioFormatMPEG4AAC
        case .alac: return kAudioFormatAppleLossless
        }
    }

    var fileType: AudioFileTypeID {
        switch self {
        case .wav: return kAudioFileWAVEType
        case .flac: return kAudioFileFLACType
        case .mp3: return kAudioFileMP3Type
        case .aac: return kAudioFileM4AType
        case .alac: return kAudioFileM4AType
        }
    }
}

enum AudioExportError: Error, LocalizedError {
    case invalidSource
    case unsupported
    case transcodingFailed(String)

    var errorDescription: String? {
        switch self {
        case .invalidSource: return "Source audio file not found."
        case .unsupported: return "Export format is not supported on this system."
        case .transcodingFailed(let reason): return "Transcoding failed: \(reason)"
        }
    }
}

struct AudioExportConfiguration: Codable, Sendable {
    var format: AudioExportFormat
    var sampleRate: Double
    var title: String
    var tags: [String]
}

final class AudioExportService: Sendable {
    func export(sourceURL: URL, destinationDirectory: URL, configuration: AudioExportConfiguration) async throws -> URL {
        guard FileManager.default.fileExists(atPath: sourceURL.path) else {
            throw AudioExportError.invalidSource
        }

        let name = "\(sanitizedFilename(configuration.title))_\(UUID().uuidString.prefix(8)).\(configuration.format.fileExtension)"
        let destination = destinationDirectory.appendingPathComponent(name)

        switch configuration.format {
        case .wav:
            try copyFile(from: sourceURL, to: destination)
            return destination
        case .flac, .alac:
            return try await transcode(
                source: sourceURL,
                destination: destination,
                format: configuration.format,
                sampleRate: configuration.sampleRate,
                bitDepth: 16
            )
        case .aac:
            return try await transcode(
                source: sourceURL,
                destination: destination,
                format: configuration.format,
                sampleRate: configuration.sampleRate,
                bitRate: 256_000
            )
        case .mp3:
            return try await transcode(
                source: sourceURL,
                destination: destination,
                format: configuration.format,
                sampleRate: configuration.sampleRate,
                bitRate: 320_000
            )
        }
    }

    private func transcode(
        source: URL,
        destination: URL,
        format: AudioExportFormat,
        sampleRate: Double,
        bitDepth: UInt32 = 16,
        bitRate: Int? = nil
    ) async throws -> URL {
        let sourceFile = try AVAudioFile(forReading: source)
        let sourceFormat = sourceFile.processingFormat

        let channelCount = sourceFormat.channelCount
        let outputSampleRate = sampleRate > 0 ? sampleRate : sourceFormat.sampleRate

        var outputSettings: [String: Any] = [
            AVFormatIDKey: format.audioFormatID,
            AVSampleRateKey: outputSampleRate,
            AVNumberOfChannelsKey: channelCount,
        ]

        switch format {
        case .wav:
            outputSettings[AVLinearPCMBitDepthKey] = bitDepth
            outputSettings[AVLinearPCMIsFloatKey] = false
            outputSettings[AVLinearPCMIsBigEndianKey] = false
            outputSettings[AVLinearPCMIsNonInterleaved] = false
        case .flac:
            break
        case .alac:
            outputSettings[AVEncoderBitDepthHintKey] = bitDepth
        case .aac:
            if let bitRate {
                outputSettings[AVEncoderBitRateKey] = bitRate
            }
            outputSettings[AVEncoderAudioQualityKey] = AVAudioQuality.max.rawValue
        case .mp3:
            if let bitRate {
                outputSettings[AVEncoderBitRateKey] = bitRate
            }
            outputSettings[AVEncoderAudioQualityKey] = AVAudioQuality.max.rawValue
        }

        removeExistingFile(at: destination)

        let asset = AVURLAsset(url: source)
        guard let assetReader = try? AVAssetReader(asset: asset) else {
            throw AudioExportError.transcodingFailed("Cannot create asset reader")
        }

        let audioTracks = try await asset.loadTracks(withMediaType: .audio)
        guard let audioTrack = audioTracks.first else {
            throw AudioExportError.transcodingFailed("No audio track in source file")
        }

        let readerOutputSettings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: outputSampleRate,
            AVNumberOfChannelsKey: channelCount,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsNonInterleaved: false,
        ]

        let readerOutput = AVAssetReaderTrackOutput(track: audioTrack, outputSettings: readerOutputSettings)
        assetReader.add(readerOutput)

        let fileType = try avFileType(for: format)
        guard let assetWriter = try? AVAssetWriter(outputURL: destination, fileType: fileType) else {
            throw AudioExportError.transcodingFailed("Cannot create asset writer")
        }

        // sourceFormatHint describes the PCM samples we feed in, not the output codec.
        // Passing the output (compressed) format here was causing AVAssetWriter to fail.
        let writerInput = AVAssetWriterInput(mediaType: .audio, outputSettings: outputSettings)
        assetWriter.add(writerInput)

        assetReader.startReading()
        guard assetWriter.startWriting() else {
            throw AudioExportError.transcodingFailed(
                assetWriter.error?.localizedDescription ?? "Cannot start writing")
        }
        assetWriter.startSession(atSourceTime: .zero)

        return try await withCheckedThrowingContinuation { continuation in
            writerInput.requestMediaDataWhenReady(on: DispatchQueue(label: "com.auralux.export")) {
                while writerInput.isReadyForMoreMediaData {
                    if let sampleBuffer = readerOutput.copyNextSampleBuffer() {
                        writerInput.append(sampleBuffer)
                    } else {
                        writerInput.markAsFinished()
                        assetWriter.finishWriting {
                            if assetWriter.status == .completed {
                                continuation.resume(returning: destination)
                            } else {
                                let msg = assetWriter.error?.localizedDescription ?? "Unknown writer error"
                                continuation.resume(throwing: AudioExportError.transcodingFailed(msg))
                            }
                        }
                        return
                    }
                }
            }
        }
    }

    private func avFileType(for format: AudioExportFormat) throws -> AVFileType {
        switch format {
        case .wav: return .wav
        case .flac:
            // AVAssetWriter does not support FLAC output — public.flac is absent from
            // its supported-UTI list and attempting to use it crashes with NSInvalidArgumentException.
            throw AudioExportError.unsupported
        case .mp3:
            // AVAssetWriter does not support MP3 encoding (licensed codec, not
            // shipped by Apple). public.mp3 is absent from its supported-UTI list.
            throw AudioExportError.unsupported
        case .aac: return .m4a
        case .alac: return .m4a
        }
    }

    private func copyFile(from source: URL, to destination: URL) throws {
        removeExistingFile(at: destination)
        try FileManager.default.copyItem(at: source, to: destination)
    }

    private func removeExistingFile(at url: URL) {
        if FileManager.default.fileExists(atPath: url.path) {
            try? FileManager.default.removeItem(at: url)
        }
    }

    private func sanitizedFilename(_ title: String) -> String {
        let illegal = CharacterSet(charactersIn: "/\\:*?\"<>|")
        let sanitized = title
            .components(separatedBy: illegal)
            .joined(separator: "_")
            .replacingOccurrences(of: " ", with: "_")
            .trimmingCharacters(in: CharacterSet(charactersIn: "_"))
        let limited = String(sanitized.prefix(80))
        return limited.isEmpty ? "track" : limited
    }
}

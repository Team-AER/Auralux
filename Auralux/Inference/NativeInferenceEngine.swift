import Foundation
import MLX
import MLXRandom
import Observation

// MARK: - Model State

enum ModelState: Equatable {
    case notDownloaded
    case downloading(progress: Double)
    case loading
    case ready
    case error(String)

    var isReady: Bool {
        if case .ready = self { return true }
        return false
    }

    var isLoading: Bool {
        switch self {
        case .loading, .downloading: return true
        default: return false
        }
    }
}

// MARK: - Generation Progress

enum GenerationProgress: Sendable {
    case preparing(message: String)
    case step(current: Int, total: Int)
    case saving
    case completed(audioURL: URL)
}

// MARK: - Errors

enum NativeEngineError: Error, LocalizedError {
    case modelsNotLoaded
    case weightsNotFound(URL)
    case generationFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelsNotLoaded:
            return "Models are not loaded. Convert weights first using tools/convert_weights.py."
        case .weightsNotFound(let url):
            return "Model weights not found at \(url.path). Run tools/convert_weights.py first."
        case .generationFailed(let message):
            return "Generation failed: \(message)"
        }
    }
}

// MARK: - Engine

@MainActor
@Observable
final class NativeInferenceEngine {

    private(set) var modelState: ModelState = .notDownloaded
    private(set) var isGenerating: Bool = false
    var isOnboarding: Bool = false

    private var dit: ACEStepDiT?
    private var lm: ACEStepLMModel?
    private var vae: DCHiFiGANDecoder?
    private var silenceLatent: MLXArray?
    private var tokenizer: BPETokenizer?
    private var textEncoder: Qwen3EncoderModel?
    private var textTokenizer: Qwen3Tokenizer?
    private var generationTask: Task<Void, Never>?
    private var activeContinuation: AsyncThrowingStream<GenerationProgress, Error>.Continuation?
    private let log = AppLogger.shared

    // MARK: - Paths

    var mlxModelDirectory: URL {
        FileUtilities.modelDirectory.appendingPathComponent("ace-step-v1.5-mlx", isDirectory: true)
    }

    var weightsExist: Bool {
        let required = [
            "dit/dit_weights.safetensors",
            "dit/silence_latent.safetensors",
            "lm/lm_weights.safetensors",
            "vae/vae_weights.safetensors",
            "text/text_weights.safetensors",
            "text/text_vocab.json",
            "text/text_merges.txt",
        ]
        return required.allSatisfy { path in
            FileManager.default.fileExists(
                atPath: mlxModelDirectory.appendingPathComponent(path).path
            )
        }
    }

    // MARK: - App Lifecycle

    func checkStatus() async {
        if weightsExist {
            guard case .notDownloaded = modelState else { return }
            await loadModels()
        } else {
            modelState = .notDownloaded
            isOnboarding = true
        }
    }

    func shutdown() {
        cancelGeneration()
        dit = nil
        lm = nil
        vae = nil
        silenceLatent = nil
        tokenizer = nil
        textEncoder = nil
        textTokenizer = nil
    }

    // MARK: - Model Download + Load

    /// Downloads all weights from HuggingFace then loads models into memory.
    /// Throws on download failure; loading errors are reflected in `modelState`.
    func downloadAndLoad() async throws {
        guard !isGenerating else { return }
        modelState = .downloading(progress: 0)
        log.info("Downloading model weights from HuggingFace", category: .inference)

        try await ModelDownloader.shared.downloadAll(to: mlxModelDirectory) { [weak self] progress in
            Task { @MainActor [weak self] in
                self?.modelState = .downloading(progress: progress)
            }
        }

        log.info("Download complete — loading models", category: .inference)
        await loadModels()
    }

    // MARK: - Model Loading

    func loadModels() async {
        guard !isGenerating else { return }
        modelState = .loading
        log.info("Loading MLX models from \(mlxModelDirectory.path)", category: .inference)

        let baseDir = mlxModelDirectory

        do {
            let models = try await Task<LoadedModels, Error>.detached(priority: .userInitiated) {
                let dit = ACEStepDiT()
                try DiTWeightLoader.load(baseDir: baseDir, into: dit)
                let silenceLatent = try SilenceLatentLoader.load(baseDir: baseDir)
                let lm = ACEStepLMModel()
                try LMWeightLoader.load(baseDir: baseDir, into: lm)
                let vae = DCHiFiGANDecoder()
                try VAEWeightLoader.load(baseDir: baseDir, into: vae)
                let tokenizer = try? BPETokenizer(
                    vocabURL: baseDir.appendingPathComponent("lm/lm_vocab.json"),
                    mergesURL: baseDir.appendingPathComponent("lm/lm_merges.txt")
                )
                let textEncoder = Qwen3EncoderModel()
                try Qwen3EncoderWeightLoader.load(baseDir: baseDir, into: textEncoder)
                let textTokenizer = try Qwen3Tokenizer.textEncoder(baseDir: baseDir)
                return LoadedModels(
                    dit: dit, lm: lm, vae: vae, silenceLatent: silenceLatent,
                    tokenizer: tokenizer, textEncoder: textEncoder, textTokenizer: textTokenizer
                )
            }.value

            self.dit = models.dit
            self.lm = models.lm
            self.vae = models.vae
            self.silenceLatent = models.silenceLatent
            self.tokenizer = models.tokenizer
            self.textEncoder = models.textEncoder
            self.textTokenizer = models.textTokenizer
            modelState = .ready
            log.info("MLX models loaded successfully", category: .inference)
        } catch {
            modelState = .error(error.localizedDescription)
            log.error("Failed to load MLX models: \(error.localizedDescription)", category: .inference)
        }
    }

    // MARK: - Generation

    func generate(request: GenerationParameters) -> AsyncThrowingStream<GenerationProgress, Error> {
        let (stream, continuation) = AsyncThrowingStream<GenerationProgress, Error>.makeStream()

        guard case .ready = modelState,
              let dit = dit, let lm = lm, let vae = vae, let silenceLatent = silenceLatent,
              let textEncoder = textEncoder, let textTokenizer = textTokenizer else {
            continuation.finish(throwing: NativeEngineError.modelsNotLoaded)
            return stream
        }

        generationTask?.cancel()
        activeContinuation?.finish(throwing: CancellationError())
        activeContinuation = continuation
        isGenerating = true

        let localCont      = continuation
        let localDit       = dit
        let localLm        = lm
        let localVae       = vae
        let localSilence   = SendableMLXArray(value: silenceLatent)
        let localTokenizer = tokenizer
        let localTextEncoder   = textEncoder
        let localTextTokenizer = textTokenizer
        let generatedDir = FileUtilities.generatedAudioDirectory
        let duration     = request.duration

        generationTask = Task.detached(priority: .userInitiated) {
            do {
                try Task.checkCancellation()
                localCont.yield(.preparing(message: "Building latent noise..."))

                // 25 Hz latent frame rate, 64-dim acoustic latent
                let T = max(1, Int(duration * 25.0))
                let acousticDim = localDit.config.audioAcousticHiddenDim
                let contextDim  = localDit.config.inChannels - acousticDim

                guard contextDim == acousticDim * 2 else {
                    throw NativeEngineError.generationFailed("Expected context dimension \(acousticDim * 2), got \(contextDim)")
                }

                let noise      = MLXRandom.normal([1, T, acousticDim])
                let srcLatents = try SilenceLatentLoader.slice(localSilence.value, frames: T)
                let chunkMasks = MLXArray.ones([1, T, acousticDim])
                let contextLatents = concatenated([srcLatents, chunkMasks], axis: -1)

                _ = localLm
                _ = localTokenizer

                // ── Build cross-attention conditioning ─────────────────────────────
                // Mirrors `AceStepConditionEncoder.forward` from upstream:
                //   text:   tokens  → Qwen3 full forward → text_projector(1024→2048)
                //   lyrics: tokens  → Qwen3 embed_tokens lookup only → lyric_encoder
                //   pack(lyric_encoded, text_projected) along the sequence dim.
                localCont.yield(.preparing(message: "Encoding conditioning..."))

                let encH = NativeInferenceEngine.buildEncoderHiddenStates(
                    request:        request,
                    dit:            localDit,
                    textEncoder:    localTextEncoder,
                    textTokenizer:  localTextTokenizer,
                    silenceLatent:  localSilence.value
                )
                eval(encH)

                try Task.checkCancellation()

                let sampler = TurboSampler()
                let result  = sampler.sample(
                    noise:               noise,
                    contextLatents:      contextLatents,
                    encoderHiddenStates: encH,
                    model:               localDit.decoder
                ) { step, total in
                    localCont.yield(.step(current: step + 1, total: total))
                }

                try Task.checkCancellation()
                localCont.yield(.saving)

                let audio = localVae.decode(latent: result)
                eval(audio)

                let filename = "generated-\(UUID().uuidString).wav"
                let outputURL = generatedDir.appendingPathComponent(filename)
                try NativeInferenceEngine.writeWAV(samples: audio, to: outputURL, sampleRate: 48000)

                try Task.checkCancellation()
                localCont.yield(.completed(audioURL: outputURL))
                localCont.finish()
            } catch {
                localCont.finish(throwing: error)
            }
        }

        continuation.onTermination = { @Sendable _ in
            Task { @MainActor [weak self] in
                self?.isGenerating = false
                self?.activeContinuation = nil
            }
        }

        return stream
    }

    func cancelGeneration() {
        generationTask?.cancel()
        generationTask = nil
        activeContinuation?.finish(throwing: CancellationError())
        activeContinuation = nil
        isGenerating = false
    }

    // MARK: - Conditioning helpers

    /// Builds the cross-attention condition tensor from a generation request.
    ///
    /// Mirrors `AceStepConditionEncoder.forward` in
    /// `modeling_acestep_v15_turbo.py:1531-1558`:
    ///   * `text_hidden_states  = text_encoder(text_ids)               → [1, S_text, 1024]`
    ///   * `text_projected      = text_projector(text_hidden_states)   → [1, S_text, 2048]`
    ///   * `lyric_hidden_states = text_encoder.embed_tokens(lyric_ids) → [1, S_lyric, 1024]`
    ///   * `lyric_encoded       = lyric_encoder(lyric_hidden_states)   → [1, S_lyric, 2048]`
    ///   * `packed              = pack(lyric_encoded, text_projected, masks)`
    ///
    /// The packed sequence with shape `[1, S_lyric+S_text, 2048]` is passed straight to the
    /// DiT cross-attention. Empty/whitespace prompts and empty lyrics fall back to upstream's
    /// learned `null_condition_emb` (the same vector seen during CFG dropout training).
    private nonisolated static func buildEncoderHiddenStates(
        request: GenerationParameters,
        dit: ACEStepDiT,
        textEncoder: Qwen3EncoderModel,
        textTokenizer: Qwen3Tokenizer,
        silenceLatent: MLXArray
    ) -> MLXArray {
        // ── Text branch (caption + tags) ─────────────────────────────────────
        let textPrompt = formatTextPrompt(prompt: request.prompt, tags: request.tags)
        let textTokens = clampTokenLength(textTokenizer.encode(textPrompt), max: 256)
        var textBranch: (hidden: MLXArray, mask: MLXArray)? = nil
        if !textTokens.isEmpty {
            let textIds = MLXArray(textTokens.map { Int32($0) }).reshaped([1, textTokens.count])
            let textHidden = textEncoder.encode(textIds)               // [1, S_text, 1024]
            let textProjected = dit.textProjector(textHidden)          // [1, S_text, 2048]
            let textMask = MLXArray.ones([1, textTokens.count]).asType(.int32)
            textBranch = (textProjected, textMask)
        }

        // ── Lyric branch (only when non-empty) ───────────────────────────────
        let lyricsRaw = request.lyrics.trimmingCharacters(in: .whitespacesAndNewlines)
        var lyricBranch: (hidden: MLXArray, mask: MLXArray)? = nil
        if !lyricsRaw.isEmpty {
            let lyricPrompt = formatLyrics(lyrics: lyricsRaw, language: request.language)
            let lyricTokens = clampTokenLength(textTokenizer.encode(lyricPrompt), max: 2048)
            if !lyricTokens.isEmpty {
                let lyricIds = MLXArray(lyricTokens.map { Int32($0) }).reshaped([1, lyricTokens.count])
                let lyricEmbeds = textEncoder.embed(lyricIds)         // [1, S_lyric, 1024]
                let lyricEncoded = dit.lyricEncoder(lyricEmbeds)      // [1, S_lyric, 2048]
                let lyricMask = MLXArray.ones([1, lyricTokens.count]).asType(.int32)
                lyricBranch = (lyricEncoded, lyricMask)
            }
        }

        // ── Timbre branch ────────────────────────────────────────────────────
        // Upstream `conditioning_batch.py:66-67` injects 30s of silence audio when
        // no reference audio is supplied; `conditioning_embed.py:46-49` then
        // substitutes the precomputed `silence_latent[:, :750, :]` slice. We mirror
        // that for text2music — no reference audio path, just silence-latent timbre.
        let timbreFrames = 750
        let timbreInput = (try? SilenceLatentLoader.slice(silenceLatent, frames: timbreFrames))
            ?? silenceLatent[0..., ..<min(timbreFrames, silenceLatent.shape[1]), 0...]
        let timbrePooled = dit.timbreEncoder(timbreInput)              // [1, hiddenSize]
        let timbreHidden = timbrePooled.reshaped([1, 1, dit.config.hiddenSize])
        let timbreMask   = MLXArray.ones([1, 1]).asType(.int32)

        // ── Pack — exactly the upstream order in AceStepConditionEncoder.forward
        // (modeling_acestep_v15_turbo.py:1556-1557): pack(lyric, timbre), then
        // pack(result, text). For text-only or lyric-only requests we still include
        // timbre — that is what upstream does for plain text2music.
        switch (lyricBranch, textBranch) {
        case let (.some(l), .some(t)):
            let (lt, ltMask) = PackSequences.pack(l.hidden, timbreHidden, l.mask, timbreMask)
            let (packed, _)  = PackSequences.pack(lt, t.hidden, ltMask, t.mask)
            return packed
        case let (.some(l), .none):
            let (packed, _) = PackSequences.pack(l.hidden, timbreHidden, l.mask, timbreMask)
            return packed
        case let (.none, .some(t)):
            let (packed, _) = PackSequences.pack(timbreHidden, t.hidden, timbreMask, t.mask)
            return packed
        case (.none, .none):
            return dit.nullConditionEmb
        }
    }

    /// Mirrors upstream `_format_lyrics` (prompt_utils.py:27-29).
    private nonisolated static func formatLyrics(lyrics: String, language: String) -> String {
        let lang = language.isEmpty ? "unknown" : language
        return "# Languages\n\(lang)\n\n# Lyric\n\(lyrics)<|endoftext|>"
    }

    /// Approximates upstream `SFT_GEN_PROMPT` (constants.py:158) using the values our
    /// `GenerationParameters` actually carries (we don't have a separate caption/metas
    /// split). Tags become `# Metas`; the prompt becomes `# Caption`.
    private nonisolated static func formatTextPrompt(prompt: String, tags: [String]) -> String {
        let caption     = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let metasJoined = tags.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                              .filter { !$0.isEmpty }
                              .joined(separator: ", ")
        let metasField  = metasJoined.isEmpty ? "" : "tags: \(metasJoined)"

        // If both empty, return empty string so the caller can fall back to nullConditionEmb.
        if caption.isEmpty && metasField.isEmpty { return "" }

        return """
        # Instruction
        Fill the audio semantic mask based on the given conditions:

        # Caption
        \(caption)

        # Metas
        \(metasField)<|endoftext|>
        """
    }

    private nonisolated static func clampTokenLength(_ ids: [Int], max: Int) -> [Int] {
        ids.count > max ? Array(ids.prefix(max)) : ids
    }

    // MARK: - WAV Export

    private nonisolated static func writeWAV(samples: MLXArray, to url: URL, sampleRate: Int) throws {
        let flat = samples.flattened()
        eval(flat)
        let floats = flat.asArray(Float.self)
        let channels = samples.shape.last == 2 ? 2 : 1
        guard floats.count % channels == 0 else {
            throw NativeEngineError.generationFailed("Audio sample count \(floats.count) is not divisible by channel count \(channels)")
        }
        let dataSize = UInt32(floats.count * 2)

        var data = Data()

        func le32(_ v: UInt32) {
            var x = v.littleEndian
            withUnsafeBytes(of: &x) { data.append(contentsOf: $0) }
        }
        func le16(_ v: UInt16) {
            var x = v.littleEndian
            withUnsafeBytes(of: &x) { data.append(contentsOf: $0) }
        }

        data.append(contentsOf: "RIFF".utf8)
        le32(36 + dataSize)
        data.append(contentsOf: "WAVE".utf8)
        data.append(contentsOf: "fmt ".utf8)
        le32(16); le16(1); le16(UInt16(channels))     // chunkSize, PCM, channels
        le32(UInt32(sampleRate))
        le32(UInt32(sampleRate * channels * 2))       // byteRate
        le16(UInt16(channels * 2)); le16(16)          // blockAlign, bitsPerSample
        data.append(contentsOf: "data".utf8)
        le32(dataSize)

        for s in floats {
            let clamped = max(-1.0, min(1.0, s))
            le16(UInt16(bitPattern: Int16(clamped * 32767)))
        }

        try data.write(to: url, options: .atomic)
    }
}

// MARK: - Private Helpers

private struct LoadedModels: @unchecked Sendable {
    let dit: ACEStepDiT
    let lm: ACEStepLMModel
    let vae: DCHiFiGANDecoder
    let silenceLatent: MLXArray
    let tokenizer: BPETokenizer?
    let textEncoder: Qwen3EncoderModel
    let textTokenizer: Qwen3Tokenizer
}

private struct SendableMLXArray: @unchecked Sendable {
    let value: MLXArray
}

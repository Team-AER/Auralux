import Foundation
import MLX
import MLXRandom
import Observation

// MARK: - Model State

enum ModelState: Equatable {
    case notDownloaded
    case downloaded         // weights on disk, not yet loaded into memory
    case downloading(progress: Double)
    case loading
    case ready
    case error(String)

    var isReady: Bool {
        switch self {
        case .ready, .downloaded: return true
        default: return false
        }
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

    /// Which variant is currently being downloaded (nil = none).
    private(set) var activeDownloadVariant: DiTVariant? = nil
    /// Overall download progress for `activeDownloadVariant` in [0, 1].
    private(set) var downloadProgress: Double = 0

    private var dit: ACEStepDiT?
    private var lm: ACEStepLMModel?
    private var vae: DCHiFiGANVAE?
    private var silenceLatent: MLXArray?
    private var tokenizer: BPETokenizer?
    private var textEncoder: Qwen3EncoderModel?
    private var textTokenizer: Qwen3Tokenizer?
    private var generationTask: Task<Void, Never>?
    private var heavyGenerationTask: Task<Void, Never>?
    private var generationID: Int = 0
    private var activeContinuation: AsyncThrowingStream<GenerationProgress, Error>.Continuation?
    private let log = AppLogger.shared

    // MARK: - Paths

    private var currentVariant: DiTVariant {
        let raw = UserDefaults.standard.string(forKey: SettingsViewModel.Keys.ditVariant) ?? "turbo"
        return DiTVariant(rawValue: raw) ?? .turbo
    }

    func mlxModelDirectory(for variant: DiTVariant) -> URL {
        FileUtilities.modelDirectory.appendingPathComponent(variant.mlxDirectoryName, isDirectory: true)
    }

    func isDownloaded(_ variant: DiTVariant) -> Bool {
        let dir = mlxModelDirectory(for: variant)
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
            FileManager.default.fileExists(atPath: dir.appendingPathComponent(path).path)
        }
    }

    var weightsExist: Bool { isDownloaded(currentVariant) }

    // MARK: - App Lifecycle

    func checkStatus() async {
        if weightsExist {
            guard case .notDownloaded = modelState else { return }
            modelState = .downloaded
        } else {
            modelState = .notDownloaded
            isOnboarding = true
        }
    }

    func shutdown() {
        cancelGeneration()
        releaseModels()
    }

    // MARK: - Model Download + Load

    /// Downloads a specific variant from HuggingFace. Non-app-downloadable variants
    /// (XL) will fail with a script instruction. SFT/base require turbo to be
    /// downloaded first (they symlink its shared components).
    func download(_ variant: DiTVariant) async {
        guard activeDownloadVariant == nil else { return }
        guard !isDownloaded(variant) else {
            if variant == currentVariant, case .notDownloaded = modelState {
                modelState = .downloaded
            }
            return
        }
        guard variant.canDownloadInApp else {
            if variant == currentVariant {
                modelState = .error(
                    "Run `python tools/convert_weights.py --variant \(variant.rawValue)` " +
                    "to convert this model (turbo must be converted first)."
                )
            }
            return
        }

        activeDownloadVariant = variant
        downloadProgress = 0
        if variant == currentVariant { modelState = .downloading(progress: 0) }
        log.info("Downloading \(variant.rawValue) weights from HuggingFace", category: .inference)

        do {
            let variantDir = mlxModelDirectory(for: variant)
            let turboDir   = mlxModelDirectory(for: .turbo)
            try await ModelDownloader.shared.download(
                variant: variant,
                to: variantDir,
                turboDirectory: turboDir
            ) { [weak self] progress in
                Task { @MainActor [weak self] in
                    self?.downloadProgress = progress
                    if variant == self?.currentVariant {
                        self?.modelState = .downloading(progress: progress)
                    }
                }
            }
            log.info("Download complete for \(variant.rawValue)", category: .inference)
        } catch {
            log.error("Download failed for \(variant.rawValue): \(error)", category: .inference)
            if variant == currentVariant {
                modelState = .error(error.localizedDescription)
            }
            activeDownloadVariant = nil
            return
        }

        activeDownloadVariant = nil
        if variant == currentVariant {
            // Discard any models from a previously loaded variant and mark ready to load on demand.
            dit = nil; lm = nil; vae = nil
            silenceLatent = nil; tokenizer = nil; textEncoder = nil; textTokenizer = nil
            MLX.Memory.clearCache()
            modelState = .downloaded
        }
    }

    /// Downloads turbo then loads — used by SetupView for first-time onboarding.
    func downloadAndLoad() async throws {
        guard !isGenerating else { return }
        await download(.turbo)
        if case .error(let msg) = modelState { throw NativeEngineError.generationFailed(msg) }
    }

    // MARK: - Model Loading

    func loadModels() async {
        switch modelState {
        case .downloaded, .error: break
        default: return
        }
        modelState = .loading

        // Read both settings at the call site: UserDefaults is thread-safe and
        // avoids capturing the @MainActor-bound SettingsViewModel in the detached task.
        let variant = currentVariant
        let baseDir = mlxModelDirectory(for: variant)
        let loadLM  = UserDefaults.standard.bool(forKey: SettingsViewModel.Keys.useLM)
        log.info("Loading MLX models (\(variant.rawValue)) from \(baseDir.path)", category: .inference)

        do {
            let models = try await Task<LoadedModels, Error>.detached(priority: .userInitiated) {
                let dit = ACEStepDiT(config: variant.modelConfig)
                try DiTWeightLoader.load(baseDir: baseDir, into: dit)
                let silenceLatent = try SilenceLatentLoader.load(baseDir: baseDir)
                // Skip LM allocation and weight load entirely when the toggle is off —
                // the LM is ~1.2 GB resident and currently unused by every code path.
                var lm: ACEStepLMModel? = nil
                if loadLM {
                    let model = ACEStepLMModel()
                    try LMWeightLoader.load(baseDir: baseDir, into: model)
                    lm = model
                }
                let vae = DCHiFiGANVAE()
                try VAEWeightLoader.load(baseDir: baseDir, into: vae)
                let tokenizer = loadLM
                    ? (try? BPETokenizer(
                        vocabURL: baseDir.appendingPathComponent("lm/lm_vocab.json"),
                        mergesURL: baseDir.appendingPathComponent("lm/lm_merges.txt")
                    ))
                    : nil
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
            // Drop any transient buffers held by safetensors loaders / one-off `eval`s
            // before we sit idle waiting for a generate request.
            MLX.Memory.clearCache()
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

        // Reject states where generation cannot even start
        switch modelState {
        case .notDownloaded:
            continuation.finish(throwing: NativeEngineError.modelsNotLoaded)
            return stream
        case .downloading:
            continuation.finish(throwing: NativeEngineError.generationFailed("A download is in progress. Please wait."))
            return stream
        case .loading:
            continuation.finish(throwing: NativeEngineError.generationFailed("Models are being loaded. Please wait."))
            return stream
        case .error(let msg):
            continuation.finish(throwing: NativeEngineError.generationFailed(msg))
            return stream
        case .downloaded, .ready:
            break
        }

        generationTask?.cancel()
        heavyGenerationTask?.cancel()
        activeContinuation?.finish(throwing: CancellationError())
        activeContinuation = continuation
        isGenerating = true
        generationID += 1
        let capturedID = generationID

        let generatedDir = FileUtilities.generatedAudioDirectory

        // Coordination task: loads models if needed, then launches heavy work.
        generationTask = Task { @MainActor [weak self] in
            guard let self else { return }

            // Phase 1: load on demand when weights are on disk but not in memory
            if case .downloaded = self.modelState {
                continuation.yield(.preparing(message: "Loading models..."))
                await self.loadModels()
                guard !Task.isCancelled else { return }
                if case .error(let msg) = self.modelState {
                    continuation.finish(
                        throwing: NativeEngineError.generationFailed("Failed to load models: \(msg)")
                    )
                    return
                }
            }

            // text2musicLM also needs the LM — surface a clean error now
            if request.mode == .text2musicLM, self.lm == nil {
                continuation.finish(
                    throwing: NativeEngineError.generationFailed(
                        "text2musicLM mode requires the 5 Hz LM. Toggle 'Load 5 Hz LM' in Settings."
                    )
                )
                return
            }

            guard case .ready = self.modelState,
                  let dit = self.dit, let vae = self.vae,
                  let silenceLatent = self.silenceLatent,
                  let textEncoder = self.textEncoder,
                  let textTokenizer = self.textTokenizer else {
                continuation.finish(throwing: NativeEngineError.modelsNotLoaded)
                return
            }

            guard !Task.isCancelled else { return }

            let localCont          = continuation
            let localDit           = dit
            let localVae           = vae
            let localLm            = self.lm
            let localSilence       = SendableMLXArray(value: silenceLatent)
            let localTextEncoder   = textEncoder
            let localTextTokenizer = textTokenizer

            // Phase 2: run the generation on a background thread
            self.heavyGenerationTask = Task.detached(priority: .userInitiated) {
                do {
                    try Task.checkCancellation()
                    localCont.yield(.preparing(message: "Preparing inputs..."))

                    let acousticDim = localDit.config.audioAcousticHiddenDim
                    let contextDim  = localDit.config.inChannels - acousticDim
                    guard contextDim == acousticDim * 2 else {
                        throw NativeEngineError.generationFailed("Expected context dimension \(acousticDim * 2), got \(contextDim)")
                    }

                    // ── Per-mode acoustic latent setup ────────────────────────────────
                    // Resolves T (latent length), src_latents, clean_src_latents (repaint),
                    // and lm_hints_25Hz (cover/text2musicLM). All shapes are [1, T, 64].
                    let inputs = try NativeInferenceEngine.prepareModeInputs(
                        request:       request,
                        dit:           localDit,
                        vae:           localVae,
                        lm:            localLm,
                        silenceLatent: localSilence.value,
                        progress: { msg in localCont.yield(.preparing(message: msg)) }
                    )
                    // Drop VAE-encoder / audio-tokenizer intermediates before we
                    // start stacking the encoder + sampler activations.
                    MLX.Memory.clearCache()
                    let T = inputs.frames
                    let srcLatentsForContext = inputs.srcLatents
                    let cleanSrcLatents = inputs.cleanSrcLatents
                    let repaintMask = inputs.repaintMask
                    let timbreLatent = inputs.timbreLatent

                    let noise      = MLXRandom.normal([1, T, acousticDim])
                    let chunkMasks = MLXArray.ones([1, T, acousticDim])
                    let contextLatents = concatenated([srcLatentsForContext, chunkMasks], axis: -1)

                    // ── Build cross-attention conditioning ─────────────────────────────
                    localCont.yield(.preparing(message: "Encoding conditioning..."))

                    let (encH, encMask) = NativeInferenceEngine.buildEncoderHiddenStates(
                        request:        request,
                        dit:            localDit,
                        textEncoder:    localTextEncoder,
                        textTokenizer:  localTextTokenizer,
                        timbreLatent:   timbreLatent
                    )
                    eval(encH, encMask)
                    // Encoder transients (Qwen3 hidden states, lyric/timbre intermediates)
                    // are no longer reachable past this point — release them before the
                    // sampler's 8 forward passes start stacking activations.
                    MLX.Memory.clearCache()

                    try Task.checkCancellation()

                    // Build sampler — choice depends on whether the loaded DiT uses
                    // CFG distillation (turbo) or expects two-pass CFG (SFT/base).
                    let result: MLXArray
                    do {
                        if localDit.config.usesCFGDistillation {
                            // Turbo: single forward pass, CFG baked into weights.
                            let sampler = try TurboSampler(
                                numSteps: request.numSteps,
                                shift:    request.scheduleShift
                            )
                            result = sampler.sample(
                                noise:                noise,
                                contextLatents:       contextLatents,
                                encoderHiddenStates:  encH,
                                encoderAttentionMask: encMask,
                                model:                localDit.decoder,
                                repaint:              cleanSrcLatents.flatMap { src in
                                    repaintMask.map { mask in
                                        TurboSampler.RepaintInputs(
                                            cleanSrcLatents:    src,
                                            mask:               mask,
                                            injectionRatio:     request.repaintInjectionRatio,
                                            crossfadeFrames:    request.repaintCrossfadeFrames,
                                            noise:              noise
                                        )
                                    }
                                }
                            ) { step, total in
                                localCont.yield(.step(current: step + 1, total: total))
                            }
                        } else {
                            // SFT / base: two forward passes per step (cond + uncond), CFG blend.
                            let sampler = try CFGSampler(
                                numSteps: request.numSteps,
                                shift:    request.scheduleShift,
                                cfgScale: Float(request.cfgScale)
                            )
                            result = sampler.sample(
                                noise:                noise,
                                contextLatents:       contextLatents,
                                encoderHiddenStates:  encH,
                                encoderAttentionMask: encMask,
                                nullConditionEmb:     localDit.nullConditionEmb,
                                model:                localDit.decoder
                            ) { step, total in
                                localCont.yield(.step(current: step + 1, total: total))
                            }
                        }
                    } catch {
                        throw NativeEngineError.generationFailed(String(describing: error))
                    }

                    try Task.checkCancellation()
                    localCont.yield(.saving)
                    // Free DiT activations before VAE decode — VAE intermediates can hit
                    // 1.5 GB+ for 60 s clips because the last Oobleck block holds [B, T*1920, 128].
                    MLX.Memory.clearCache()

                    let audio = localVae.decode(latent: result)
                    eval(audio)
                    // Release VAE intermediates now that we have the final waveform.
                    MLX.Memory.clearCache()

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
        }

        continuation.onTermination = { @Sendable [capturedID] _ in
            Task { @MainActor [weak self, capturedID] in
                guard let self, self.generationID == capturedID else { return }
                self.isGenerating = false
                self.activeContinuation = nil
                self.heavyGenerationTask = nil
                self.releaseModels()
            }
        }

        return stream
    }

    func cancelGeneration() {
        generationTask?.cancel()
        heavyGenerationTask?.cancel()
        generationTask = nil
        heavyGenerationTask = nil
        activeContinuation?.finish(throwing: CancellationError())
        activeContinuation = nil
        isGenerating = false
    }

    private func releaseModels() {
        dit = nil; lm = nil; vae = nil
        silenceLatent = nil; tokenizer = nil; textEncoder = nil; textTokenizer = nil
        MLX.Memory.clearCache()
        if case .ready = modelState { modelState = .downloaded }
    }

    // MARK: - Mode-specific input preparation

    /// Result of `prepareModeInputs`. Resolves T (latent length) and the four
    /// 25 Hz acoustic latents that mode-aware generation needs, plus the
    /// repaint mask for sampler-time injection.
    private struct ModeInputs {
        let frames: Int
        let srcLatents: MLXArray             // [1, T, 64] — feeds into `concat(_, chunk_masks)`
        let cleanSrcLatents: MLXArray?       // [1, T, 64] — only set for repaint
        let repaintMask: MLXArray?           // [1, T] bool — only set for repaint
        let timbreLatent: MLXArray           // [1, 750, 64] — fed into `dit.timbreEncoder`
    }


    /// Resolve `srcLatents`, `cleanSrcLatents`, `repaintMask`, and `timbreLatent`
    /// per the requested mode. Mirrors the relevant branches in upstream
    /// `AceStepConditionGenerationModel.generate_audio` /
    /// `AceStepConditionGenerationModel.prepare_condition` —
    /// `modeling_acestep_v15_turbo.py:1654-1699` for the cover/audio_codes path
    /// and `:2178-2187` for the repaint mask path.
    private nonisolated static func prepareModeInputs(
        request: GenerationParameters,
        dit: ACEStepDiT,
        vae: DCHiFiGANVAE,
        lm: ACEStepLMModel?,
        silenceLatent: MLXArray,
        progress: (String) -> Void
    ) throws -> ModeInputs {
        let mode = request.mode

        // Default timbre = silence-latent slice (used by text2music + LM modes).
        func defaultTimbre() throws -> MLXArray {
            (try? SilenceLatentLoader.slice(silenceLatent, frames: kTimbreFrames))
                ?? silenceLatent[0..., ..<min(kTimbreFrames, silenceLatent.shape[1]), 0...]
        }

        // Helper: Encode a user-provided audio file → 25 Hz, 64-dim latent.
        // Returns `[1, T, 64]`.
        func encodeUserAudio(_ url: URL) throws -> MLXArray {
            let audio = try AudioFileLoader.load(url: url)            // [1, T_audio, 2]
            let latent = vae.encode(audio: audio)                      // [1, T_audio/1920, 64]
            eval(latent)
            return latent
        }

        switch mode {

        // ── text2music: silence-latent for context AND timbre ─────────────────
        case .text2music:
            let T = max(1, Int(request.duration * kFrameRateHz))
            let src = try SilenceLatentLoader.slice(silenceLatent, frames: T)
            return ModeInputs(
                frames:           T,
                srcLatents:       src,
                cleanSrcLatents:  nil,
                repaintMask:      nil,
                timbreLatent:     try defaultTimbre()
            )

        // ── extract: refer-audio timbre, silence-latent context ───────────────
        case .extract:
            guard let referURL = request.referAudioURL else {
                throw NativeEngineError.generationFailed("Extract mode requires a reference audio file.")
            }
            progress("Encoding reference audio...")
            let referLatent = try encodeUserAudio(referURL)
            let T = max(1, Int(request.duration * kFrameRateHz))
            let src = try SilenceLatentLoader.slice(silenceLatent, frames: T)
            // Tile or trim to the canonical 750 timbre frames.
            let timbre = try sliceOrTile(referLatent, frames: kTimbreFrames, fallback: silenceLatent)
            return ModeInputs(
                frames:           T,
                srcLatents:       src,
                cleanSrcLatents:  nil,
                repaintMask:      nil,
                timbreLatent:     timbre
            )

        // ── cover: source-audio context (passed through audio tokenizer for
        // 5 Hz semantic distillation), refer-audio timbre if provided.
        case .cover:
            guard let sourceURL = request.sourceAudioURL else {
                throw NativeEngineError.generationFailed("Cover mode requires a source audio file.")
            }
            progress("Encoding source audio...")
            let sourceLatent = try encodeUserAudio(sourceURL)          // [1, T, 64]
            // Round T down to a pool_window_size multiple (5 frames @ 25 Hz = 200 ms).
            // This matches upstream's pad-then-tokenize loop in `tokenize(...)`.
            let pool = dit.config.poolWindowSize
            let T0 = sourceLatent.shape[1]
            let T  = (T0 / pool) * pool
            guard T > 0 else {
                throw NativeEngineError.generationFailed("Source audio too short for cover (need ≥ \(pool) latent frames).")
            }
            let trimmed = sourceLatent[0..., ..<T, 0...]
            // Tokenize → detokenize to produce the LM-hint stand-in for `src_latents`.
            progress("Distilling source audio...")
            let (quantized, _) = dit.audioTokenizer(trimmed)            // [1, T/5, hidden]
            let lmHints25 = dit.detokenizer(quantized)                  // [1, T, 64]
            eval(lmHints25)

            // Timbre: use refer audio if provided, else silence.
            let timbre: MLXArray
            if let referURL = request.referAudioURL {
                progress("Encoding reference audio...")
                let referLatent = try encodeUserAudio(referURL)
                timbre = try sliceOrTile(referLatent, frames: kTimbreFrames, fallback: silenceLatent)
            } else {
                timbre = try defaultTimbre()
            }
            return ModeInputs(
                frames:           T,
                srcLatents:       lmHints25,
                cleanSrcLatents:  nil,
                repaintMask:      nil,
                timbreLatent:     timbre
            )

        // ── repaint: source-audio context as `clean_src_latents`; mask drives
        // sampler injection; non-mask regions kept from source.
        case .repaint:
            guard let sourceURL = request.sourceAudioURL else {
                throw NativeEngineError.generationFailed("Repaint mode requires a source audio file.")
            }
            progress("Encoding source audio...")
            let sourceLatent = try encodeUserAudio(sourceURL)          // [1, T, 64]
            let T = sourceLatent.shape[1]
            guard T > 0 else {
                throw NativeEngineError.generationFailed("Source audio decoded to zero latent frames.")
            }
            let mask = repaintMaskTensor(ranges: request.repaintMaskRanges, frames: T)
            // For the context concat, keep the *source* latents — they get
            // selectively replaced inside the sampler at non-repaint frames.
            return ModeInputs(
                frames:           T,
                srcLatents:       sourceLatent,
                cleanSrcLatents:  sourceLatent,
                repaintMask:      mask,
                timbreLatent:     try defaultTimbre()
            )

        // ── text2musicLM: LM autoregressively generates audio_codes → 5 Hz hints
        case .text2musicLM:
            guard let lm else {
                throw NativeEngineError.generationFailed("text2musicLM mode requires the 5 Hz LM toggle to be on.")
            }
            let T = max(1, Int(request.duration * kFrameRateHz))
            // Pool window must divide T (audio codes are at 5 Hz).
            let pool = dit.config.poolWindowSize
            let aligned = (T / pool) * pool
            guard aligned > 0 else {
                throw NativeEngineError.generationFailed("Duration too short for LM mode (need ≥ \(Double(pool) / kFrameRateHz)s).")
            }
            progress("Generating audio codes (LM)...")
            let codeFrames = aligned / pool
            // Conditioning prompt for the LM: re-use the same caption text as the DiT.
            let promptText = formatTextPrompt(
                prompt: request.prompt, tags: request.tags, duration: request.duration
            )
            let codes = try ACEStepLMSampler.generate(
                lm:            lm,
                prompt:        promptText,
                lyrics:        request.lyrics,
                language:      request.language,
                codeFrames:    codeFrames,
                seed:          request.seed
            )
            // codes: [1, codeFrames, num_quantizers] — turn back into 25 Hz hints.
            let pooled = dit.audioTokenizer.quantizer.getOutputFromIndices(codes)  // [1, T_5, hidden]
            let hints25 = dit.detokenizer(pooled)                                  // [1, T, 64]
            eval(hints25)
            return ModeInputs(
                frames:           aligned,
                srcLatents:       hints25,
                cleanSrcLatents:  nil,
                repaintMask:      nil,
                timbreLatent:     try defaultTimbre()
            )
        }
    }

    /// Tile a short latent or trim a long one to exactly `frames` frames along
    /// the time axis. Falls back to `fallback`'s leading slice when the encoder
    /// produced an empty array (shouldn't happen, but safer than crashing).
    private nonisolated static func sliceOrTile(
        _ latent: MLXArray, frames: Int, fallback: MLXArray
    ) throws -> MLXArray {
        let T = latent.shape[1]
        guard T > 0 else {
            return try SilenceLatentLoader.slice(fallback, frames: frames)
        }
        if T >= frames {
            return latent[0..., ..<frames, 0...]
        }
        let repeats = (frames + T - 1) / T
        let tiled = concatenated(Array(repeating: latent, count: repeats), axis: 1)
        return tiled[0..., ..<frames, 0...]
    }

    /// Build a `[1, T]` boolean mask (as int32 0/1) from a list of repaint
    /// ranges. Ranges are clamped to `[0, T)` and overlapping ranges are
    /// unioned. An empty list returns an all-zero mask (= regenerate nothing,
    /// effectively a no-op repaint). An out-of-bounds-only list also yields
    /// all-zero — the caller should warn but the engine doesn't fail.
    private nonisolated static func repaintMaskTensor(
        ranges: [RepaintRange], frames: Int
    ) -> MLXArray {
        var bits = [Int32](repeating: 0, count: frames)
        for r in ranges {
            let s = max(0, Int((r.start * kFrameRateHz).rounded(.down)))
            let e = min(frames, Int((r.end * kFrameRateHz).rounded(.up)))
            guard e > s else { continue }
            for i in s..<e { bits[i] = 1 }
        }
        return MLXArray(bits, [1, frames])
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
        timbreLatent: MLXArray
    ) -> (hidden: MLXArray, mask: MLXArray) {
        // ── Text branch (caption + tags) ─────────────────────────────────────
        let textPrompt = formatTextPrompt(
            prompt: request.prompt, tags: request.tags, duration: request.duration
        )
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
        // no reference audio is supplied; `conditioning_embed.py:46-49` substitutes
        // `silence_latent[:, :750, :]`. For Extract / Cover we instead pass in the
        // VAE-encoded reference audio's first 750 frames (i.e. `timbreLatent` from
        // `prepareModeInputs`).
        let timbrePooled = dit.timbreEncoder(timbreLatent)             // [1, encoderHiddenSize]
        let timbreHidden = timbrePooled.reshaped([1, 1, dit.config.encoderHiddenSize])
        let timbreMask   = MLXArray.ones([1, 1]).asType(.int32)

        // ── Pack — exactly the upstream order in AceStepConditionEncoder.forward
        // (modeling_acestep_v15_turbo.py:1556-1557): pack(lyric, timbre), then
        // pack(result, text). For text-only or lyric-only requests we still include
        // timbre — that is what upstream does for plain text2music.
        // We retain the packed key-padding mask for the DiT cross-attention
        // (modeling_acestep_v15_turbo.py:516).
        switch (lyricBranch, textBranch) {
        case let (.some(l), .some(t)):
            let (lt, ltMask)  = PackSequences.pack(l.hidden, timbreHidden, l.mask, timbreMask)
            let (packed, mk)  = PackSequences.pack(lt, t.hidden, ltMask, t.mask)
            return (packed, mk)
        case let (.some(l), .none):
            let (packed, mk)  = PackSequences.pack(l.hidden, timbreHidden, l.mask, timbreMask)
            return (packed, mk)
        case let (.none, .some(t)):
            let (packed, mk)  = PackSequences.pack(timbreHidden, t.hidden, timbreMask, t.mask)
            return (packed, mk)
        case (.none, .none):
            // null_condition_emb is `[1, 1, hiddenSize]` — single valid position.
            let mk = MLXArray.ones([1, dit.nullConditionEmb.shape[1]]).asType(.int32)
            return (dit.nullConditionEmb, mk)
        }
    }

    /// Mirrors upstream `_format_lyrics` (prompt_utils.py:27-29).
    private nonisolated static func formatLyrics(lyrics: String, language: String) -> String {
        let lang = language.isEmpty ? "unknown" : language
        return "# Languages\n\(lang)\n\n# Lyric\n\(lyrics)<|endoftext|>"
    }

    /// Builds the `SFT_GEN_PROMPT` text input exactly as upstream
    /// (`acestep/constants.py:101-109` + `acestep/handler.py:_dict_to_meta_string` 920-944).
    ///
    /// Upstream `# Metas` is a structured key-value block, NOT a tag list:
    /// ```
    /// - bpm: <bpm or N/A>
    /// - timesignature: <ts or N/A>
    /// - keyscale: <ks or N/A>
    /// - duration: <int> seconds
    /// ```
    /// The model was SFT-trained on this exact form. We currently don't carry
    /// bpm/timesignature/keyscale through `GenerationParameters`, so they default to
    /// "N/A" (matching `_create_default_meta`). Tags are stylistic descriptors and
    /// belong with the caption — append them there rather than the metas block.
    private nonisolated static func formatTextPrompt(
        prompt: String,
        tags: [String],
        duration: TimeInterval
    ) -> String {
        var caption = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let tagsJoined = tags
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: ", ")
        if !tagsJoined.isEmpty {
            caption = caption.isEmpty ? tagsJoined : "\(caption), \(tagsJoined)"
        }
        if caption.isEmpty { return "" }   // caller falls back to nullConditionEmb

        let durStr = "\(Int(duration)) seconds"
        return """
        # Instruction
        Fill the audio semantic mask based on the given conditions:

        # Caption
        \(caption)

        # Metas
        - bpm: N/A
        - timesignature: N/A
        - keyscale: N/A
        - duration: \(durStr)<|endoftext|>
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

// Latent frame rate (25 Hz, hardcoded by the v1.5 turbo VAE) and the
// canonical timbre slice length used by `AceStepTimbreEncoder`. Both are
// nonisolated so the detached generation task can read them safely.
private let kFrameRateHz: Double = 25
private let kTimbreFrames: Int = 750

private struct LoadedModels: @unchecked Sendable {
    let dit: ACEStepDiT
    let lm: ACEStepLMModel?
    let vae: DCHiFiGANVAE
    let silenceLatent: MLXArray
    let tokenizer: BPETokenizer?
    let textEncoder: Qwen3EncoderModel
    let textTokenizer: Qwen3Tokenizer
}

private struct SendableMLXArray: @unchecked Sendable {
    let value: MLXArray
}

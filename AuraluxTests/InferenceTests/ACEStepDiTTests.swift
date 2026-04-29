import XCTest
import MLX
import MLXNN
import MLXRandom
@testable import Auralux

/// Shape tests for ACE-Step v1.5 Turbo DiT and related components.
/// Uses a mini config (hiddenSize=64, 2 layers) for fast execution without real weights.
/// Run via Xcode ⌘U — MLX requires Metal; `swift test` CLI will fail.
final class ACEStepDiTTests: XCTestCase {

    // Mini config with consistent small dimensions: numHeads * headDim == hiddenSize.
    private let miniConfig = AceStepConfig(
        hiddenSize:               64,
        numHeads:                 4,
        numKVHeads:               2,
        headDim:                  16,
        intermediateSize:         128,
        numDiTLayers:             2,
        numLyricEncoderLayers:    2,
        numDetokenizerLayers:     1,
        patchSize:                2,
        poolWindowSize:           3,
        audioAcousticHiddenDim:   8,
        inChannels:               24,   // contextDim(16) + acousticDim(8)
        rmsNormEps:               1e-6,
        ropeTheta:                10_000.0,
        textHiddenDim:            16,
        freqDim:                  32
    )

    // MARK: - Config

    func testDefaultConfigValues() {
        let cfg = AceStepConfig()
        XCTAssertEqual(cfg.hiddenSize,              2048)
        XCTAssertEqual(cfg.numHeads,                16)
        XCTAssertEqual(cfg.numKVHeads,              8)
        XCTAssertEqual(cfg.headDim,                 128)
        XCTAssertEqual(cfg.numDiTLayers,            24)
        XCTAssertEqual(cfg.audioAcousticHiddenDim,  64)
        XCTAssertEqual(cfg.inChannels,              192)
        XCTAssertEqual(cfg.textHiddenDim,           1024)
        XCTAssertEqual(cfg.freqDim,                 256)
    }

    // MARK: - SwiGLU MLP

    func testSwiGLUMLPShape() {
        let mlp = SwiGLUMLP(hiddenSize: 64, intermediateSize: 128)
        let x   = MLXRandom.normal([1, 8, 64])
        let out = mlp(x)
        eval(out)
        XCTAssertEqual(out.shape, [1, 8, 64])
        XCTAssertTrue(out.sum().item(Float.self).isFinite)
    }

    // MARK: - Timestep Embedder

    func testTimestepEmbedderShape() {
        let emb         = TimestepEmbedder(freqDim: 32, hiddenSize: 64)
        let t           = MLXArray([0.5, 0.25] as [Float])
        let (temb, proj) = emb(t)
        eval(temb); eval(proj)
        XCTAssertEqual(temb.shape, [2, 64])
        XCTAssertEqual(proj.shape, [2, 6, 64])
        XCTAssertTrue(temb.sum().item(Float.self).isFinite)
    }

    // MARK: - Attention (GQA)

    func testSelfAttentionShape() {
        let cfg  = miniConfig
        let attn = AceStepAttention(
            hiddenSize: cfg.hiddenSize, numHeads: cfg.numHeads,
            numKVHeads: cfg.numKVHeads, headDim: cfg.headDim,
            rmsNormEps: cfg.rmsNormEps, ropeTheta: cfg.ropeTheta
        )
        let x   = MLXRandom.normal([1, 10, cfg.hiddenSize])
        let out = attn(x)
        eval(out)
        XCTAssertEqual(out.shape, [1, 10, cfg.hiddenSize])
        XCTAssertTrue(out.sum().item(Float.self).isFinite)
    }

    func testCrossAttentionShape() {
        let cfg  = miniConfig
        let attn = AceStepAttention(
            hiddenSize: cfg.hiddenSize, numHeads: cfg.numHeads,
            numKVHeads: cfg.numKVHeads, headDim: cfg.headDim,
            rmsNormEps: cfg.rmsNormEps, ropeTheta: cfg.ropeTheta,
            isCrossAttention: true
        )
        let x   = MLXRandom.normal([1, 10, cfg.hiddenSize])
        let enc = MLXRandom.normal([1,  5, cfg.hiddenSize])
        let out = attn(x, encoderHidden: enc)
        eval(out)
        XCTAssertEqual(out.shape, [1, 10, cfg.hiddenSize])
    }

    func testAttentionBatched() {
        let cfg  = miniConfig
        let attn = AceStepAttention(
            hiddenSize: cfg.hiddenSize, numHeads: cfg.numHeads,
            numKVHeads: cfg.numKVHeads, headDim: cfg.headDim,
            rmsNormEps: cfg.rmsNormEps, ropeTheta: cfg.ropeTheta
        )
        let x   = MLXRandom.normal([2, 16, cfg.hiddenSize])
        let out = attn(x)
        eval(out)
        XCTAssertEqual(out.shape, [2, 16, cfg.hiddenSize])
    }

    // MARK: - DiT Layer (AdaLN-Zero)

    func testDiTLayerShape() {
        let layer        = AceStepDiTLayer(config: miniConfig)
        let B = 1, T = 8
        let dim          = miniConfig.hiddenSize
        let x            = MLXRandom.normal([B, T, dim])
        let timestepProj = MLXRandom.normal([B, 6, dim])
        let enc          = MLXRandom.normal([B, 5, dim])
        let out          = layer(x, timestepProj: timestepProj, encoderHidden: enc)
        eval(out)
        XCTAssertEqual(out.shape, [B, T, dim])
        XCTAssertTrue(out.sum().item(Float.self).isFinite)
    }

    // MARK: - Lyric Encoder

    func testLyricEncoderShape() {
        let enc    = AceLyricEncoder(config: miniConfig)
        // Input: pre-computed text embeddings [B, S, textHiddenDim]
        let embeds = MLXRandom.normal([1, 10, miniConfig.textHiddenDim])
        let out    = enc(embeds)
        eval(out)
        XCTAssertEqual(out.shape, [1, 10, miniConfig.hiddenSize])
        XCTAssertTrue(out.sum().item(Float.self).isFinite)
    }

    // MARK: - Audio Detokenizer

    func testDetokenizerShape() {
        let det  = AceAudioDetokenizer(config: miniConfig)
        let P    = miniConfig.poolWindowSize
        // Input: [B, T_tok, hiddenSize]
        let x    = MLXRandom.normal([1, 6, miniConfig.hiddenSize])
        let out  = det(x)
        eval(out)
        // Output: [B, T_tok * poolWindowSize, audioAcousticHiddenDim]
        XCTAssertEqual(out.shape, [1, 6 * P, miniConfig.audioAcousticHiddenDim])
        XCTAssertTrue(out.sum().item(Float.self).isFinite)
    }

    // MARK: - DiT Model (decoder)

    func testDiTModelForwardShape() {
        let model       = AceStepDiTModel(config: miniConfig)
        let B = 1, T = 8
        let acousticDim = miniConfig.audioAcousticHiddenDim          // 8
        let contextDim  = miniConfig.inChannels - acousticDim         // 16
        let hiddenSize  = miniConfig.hiddenSize                        // 64

        let out = model.callAsFunction(
            hiddenStates:        MLXRandom.normal([B, T, acousticDim]),
            contextLatents:      MLXArray.zeros([B, T, contextDim]),
            timestep:            MLXArray([0.5] as [Float]),
            timestepR:           MLXArray([0.5] as [Float]),
            encoderHiddenStates: MLXArray.zeros([B, 1, hiddenSize])
        )
        eval(out)
        XCTAssertEqual(out.shape, [B, T, acousticDim])
        XCTAssertTrue(out.sum().item(Float.self).isFinite)
    }

    func testDiTModelPadsThenCrops() {
        // T=5 is not divisible by patchSize=2 — model should pad internally then crop
        let model       = AceStepDiTModel(config: miniConfig)
        let T           = 5
        let acousticDim = miniConfig.audioAcousticHiddenDim
        let contextDim  = miniConfig.inChannels - acousticDim

        let out = model.callAsFunction(
            hiddenStates:        MLXRandom.normal([1, T, acousticDim]),
            contextLatents:      MLXArray.zeros([1, T, contextDim]),
            timestep:            MLXArray([0.5] as [Float]),
            timestepR:           MLXArray([0.5] as [Float]),
            encoderHiddenStates: MLXArray.zeros([1, 1, miniConfig.hiddenSize])
        )
        eval(out)
        XCTAssertEqual(out.shape, [1, T, acousticDim])
    }

    // MARK: - TurboSampler

    func testTurboSamplerDefaultSchedule() {
        let sampler = TurboSampler()
        XCTAssertEqual(sampler.numSteps, 8)
        XCTAssertEqual(sampler.schedule.first!, 1.0,    accuracy: 1e-6)
        XCTAssertEqual(sampler.schedule.last!,  0.125,  accuracy: 1e-6)
        // Schedule must be strictly descending
        for i in 1..<sampler.schedule.count {
            XCTAssertLessThan(sampler.schedule[i], sampler.schedule[i - 1])
        }
    }

    func testTurboSamplerOutputShape() throws {
        let sampler     = TurboSampler(schedule: [1.0, 0.5])   // 2 steps for speed
        let model       = AceStepDiTModel(config: miniConfig)
        let B = 1, T = 4
        let acousticDim = miniConfig.audioAcousticHiddenDim
        let contextDim  = miniConfig.inChannels - acousticDim

        var stepCount = 0
        let result = try sampler.sample(
            noise:               MLXRandom.normal([B, T, acousticDim]),
            contextLatents:      MLXArray.zeros([B, T, contextDim]),
            encoderHiddenStates: MLXArray.zeros([B, 1, miniConfig.hiddenSize]),
            model:               model
        ) { _, _ in stepCount += 1 }
        eval(result)

        XCTAssertEqual(result.shape, [B, T, acousticDim])
        XCTAssertTrue(result.sum().item(Float.self).isFinite)
        XCTAssertEqual(stepCount, 2)
    }

    // MARK: - Audio VAE + Context

    func testAudioVAEDecoderShape() {
        let vae   = DCHiFiGANDecoder()
        let audio = vae.decode(latent: MLXArray.zeros([1, 1, 64]))
        eval(audio)
        XCTAssertEqual(audio.shape, [1, 1 * 1920, 2])
    }

    func testSilenceLatentSliceTilesToRequestedFrames() throws {
        let latent = MLXArray.zeros([1, 3, 64])
        let sliced = try SilenceLatentLoader.slice(latent, frames: 8)
        eval(sliced)
        XCTAssertEqual(sliced.shape, [1, 8, 64])
    }
}

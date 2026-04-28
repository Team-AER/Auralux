import XCTest
import MLX
import MLXNN
import MLXRandom
@testable import Auralux

/// Phase 2 shape tests for the ACE-Step 5Hz Language Model (Qwen2-style GQA).
/// These run without real weights — they verify architecture dimensions only.
/// Run via Xcode ⌘U (not `swift test`: Metal/MLX requires GPU access).
final class ACEStepLMTests: XCTestCase {

    // Minimal config for fast tests (fewer layers, smaller hidden size)
    private let miniConfig = ACEStepLMConfig(
        vocabSize: 256,
        hiddenSize: 64,
        numHiddenLayers: 2,
        numAttentionHeads: 4,
        numKeyValueHeads: 2,
        intermediateSize: 128,
        rmsNormEps: 1e-6,
        ropeTheta: 10_000.0,
        maxPositionEmbeddings: 512
    )

    // MARK: - Config

    func testDefaultConfigValues() {
        let cfg = ACEStepLMConfig()
        XCTAssertEqual(cfg.hiddenSize, 1024)
        XCTAssertEqual(cfg.numAttentionHeads, 16)
        XCTAssertEqual(cfg.numKeyValueHeads, 8)
        XCTAssertEqual(cfg.numHiddenLayers, 28)
        XCTAssertEqual(cfg.vocabSize, 217204)
    }

    // MARK: - Attention

    func testAttentionOutputShape() {
        let attn = ACEStepLMAttention(config: miniConfig)
        let x = MLXRandom.normal([1, 10, miniConfig.hiddenSize])
        let out = attn(x)
        eval(out)
        XCTAssertEqual(out.shape, [1, 10, miniConfig.hiddenSize])
    }

    func testAttentionWithCausalMask() {
        let attn = ACEStepLMAttention(config: miniConfig)
        let x = MLXRandom.normal([2, 8, miniConfig.hiddenSize])
        let out = attn(x)
        eval(out)
        XCTAssertEqual(out.shape, [2, 8, miniConfig.hiddenSize])
        XCTAssertTrue(out.sum().item(Float.self).isFinite)
    }

    func testGQAHeadRatio() {
        // Verify GQA (numKVHeads=2, numHeads=4 → g=2) produces correct output
        let attn = ACEStepLMAttention(config: miniConfig)
        let x = MLXRandom.normal([1, 5, miniConfig.hiddenSize])
        let out = attn(x)
        eval(out)
        XCTAssertEqual(out.shape[2], miniConfig.hiddenSize)
    }

    // MARK: - FFN

    func testFFNOutputShape() {
        let ffn = ACEStepLMFFN(config: miniConfig)
        let x = MLXRandom.normal([2, 10, miniConfig.hiddenSize])
        let out = ffn(x)
        eval(out)
        XCTAssertEqual(out.shape, [2, 10, miniConfig.hiddenSize])
    }

    // MARK: - Block

    func testBlockOutputShape() {
        let block = ACEStepLMBlock(config: miniConfig)
        let x = MLXRandom.normal([1, 12, miniConfig.hiddenSize])
        let out = block(x)
        eval(out)
        XCTAssertEqual(out.shape, [1, 12, miniConfig.hiddenSize])
    }

    func testBlockResidualFinite() {
        let block = ACEStepLMBlock(config: miniConfig)
        let x = MLXRandom.normal([1, 6, miniConfig.hiddenSize])
        let out = block(x)
        eval(out)
        XCTAssertTrue(out.sum().item(Float.self).isFinite)
    }

    // MARK: - Full model

    func testModelForwardShape() {
        let model = ACEStepLMModel(config: miniConfig)
        let ids = MLXArray.zeros([1, 8], dtype: .int32)
        let logits = model(ids)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 8, miniConfig.vocabSize])
    }

    func testModelForwardBatched() {
        let model = ACEStepLMModel(config: miniConfig)
        let ids = MLXArray.zeros([2, 6], dtype: .int32)
        let logits = model(ids)
        eval(logits)
        XCTAssertEqual(logits.shape, [2, 6, miniConfig.vocabSize])
        XCTAssertTrue(logits.sum().item(Float.self).isFinite)
    }

    // MARK: - Generation

    func testGenerateProducesTokens() {
        let model = ACEStepLMModel(config: miniConfig)
        let tokens = model.generate(inputIds: [0, 1, 2], maxNewTokens: 4, temperature: 1.0)
        XCTAssertEqual(tokens.count, 4)
        for t in tokens {
            XCTAssertGreaterThanOrEqual(t, 0)
            XCTAssertLessThan(t, miniConfig.vocabSize)
        }
    }

    func testGenerateGreedy() {
        let model = ACEStepLMModel(config: miniConfig)
        let t1 = model.generate(inputIds: [0], maxNewTokens: 3, temperature: 0.0)
        let t2 = model.generate(inputIds: [0], maxNewTokens: 3, temperature: 0.0)
        // Greedy decoding is deterministic given the same weights
        XCTAssertEqual(t1, t2)
    }
}

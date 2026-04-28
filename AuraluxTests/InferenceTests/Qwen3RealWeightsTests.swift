import XCTest
import MLX
import MLXNN
@testable import Auralux

/// Integration tests that load the *actual* converted Qwen3-Embedding-0.6B
/// weights and run forward passes. Skipped when weights are not present.
///
/// These are slower (loading is ~1.2 GB) so they live in a dedicated test class
/// — keep `Qwen3ConditioningTests` (unit) fast.
final class Qwen3RealWeightsTests: XCTestCase {

    private static let modelBaseDir: URL = FileUtilities.modelDirectory
        .appendingPathComponent("ace-step-v1.5-mlx", isDirectory: true)

    private func skipIfMissing() throws {
        let p = Self.modelBaseDir.appendingPathComponent("text/text_weights.safetensors")
        guard FileManager.default.fileExists(atPath: p.path) else {
            throw XCTSkip("Text-encoder weights missing — run tools/convert_weights.py")
        }
    }

    // MARK: - Loading

    func testTextEncoderWeightsLoad() throws {
        try skipIfMissing()
        let model = Qwen3EncoderModel()
        try Qwen3EncoderWeightLoader.load(baseDir: Self.modelBaseDir, into: model)

        // Spot-check a known parameter: embedTokens for vocab 151669 × 1024.
        let embedShape = model.embedTokens.weight.shape
        XCTAssertEqual(embedShape, [151669, 1024],
                       "Qwen3-Embedding embed_tokens shape should be [151669, 1024]")
    }

    // MARK: - Embed lookup (lyric path)

    func testEmbedLookupOnHelloProducesNonZero() throws {
        try skipIfMissing()
        let model = Qwen3EncoderModel()
        try Qwen3EncoderWeightLoader.load(baseDir: Self.modelBaseDir, into: model)

        let tok = try Qwen3Tokenizer.textEncoder(baseDir: Self.modelBaseDir)
        let ids = tok.encode("Hello")
        // `encode` appends 151643 at the end to mirror upstream's tokenizer __call__.
        XCTAssertEqual(ids, [9707, 151643])

        let inputIds = MLXArray(ids.map { Int32($0) }).reshaped([1, ids.count])
        let emb = model.embed(inputIds)
        eval(emb)

        XCTAssertEqual(emb.shape, [1, 2, 1024])
        let absMean = abs(emb).mean().item(Float.self)
        XCTAssertGreaterThan(absMean, 1e-4,
                             "Embed lookup should produce non-degenerate values")
        XCTAssertTrue(emb.sum().item(Float.self).isFinite, "Output must be finite")
    }

    // MARK: - Full encode (text path)

    func testEncodeFullForwardOnPromptProducesNonZero() throws {
        try skipIfMissing()
        let model = Qwen3EncoderModel()
        try Qwen3EncoderWeightLoader.load(baseDir: Self.modelBaseDir, into: model)

        let tok = try Qwen3Tokenizer.textEncoder(baseDir: Self.modelBaseDir)
        // Realistic short prompt — keep small to keep this test fast.
        let ids = tok.encode("chill lofi piano")
        XCTAssertGreaterThan(ids.count, 1)

        let inputIds = MLXArray(ids.map { Int32($0) }).reshaped([1, ids.count])
        let h = model.encode(inputIds)
        eval(h)

        XCTAssertEqual(h.shape, [1, ids.count, 1024])
        let absMean = abs(h).mean().item(Float.self)
        XCTAssertGreaterThan(absMean, 1e-4,
                             "Full encode should produce non-degenerate hidden states")
        XCTAssertTrue(h.sum().item(Float.self).isFinite, "Output must be finite")

        // Sanity: encode output should differ from raw embed lookup
        // (otherwise the 28 transformer layers did nothing).
        let emb = model.embed(inputIds)
        eval(emb)
        let diff = abs(h - emb).mean().item(Float.self)
        XCTAssertGreaterThan(diff, 1e-3,
                             "encode() output should differ meaningfully from embed-only lookup")
    }

    // MARK: - End-to-end conditioning

    /// Regression test for the fp16 overflow bug in `lyric_encoder` MLP/RMSNorm
    /// that produced NaN on lyric prompts longer than ~30 tokens. Upstream Python
    /// runs in fp32 — Swift now upcasts inside the lyric_encoder body to match.
    /// Without the fix, this test produces NaN; with the fix, the output is finite
    /// and matches Python's last_hidden_state range of approximately ±10.
    func testLyricEncoderHandlesLongLyricsWithoutNaN() throws {
        try skipIfMissing()
        let dit = ACEStepDiT()
        try DiTWeightLoader.load(baseDir: Self.modelBaseDir, into: dit)
        let textEncoder = Qwen3EncoderModel()
        try Qwen3EncoderWeightLoader.load(baseDir: Self.modelBaseDir, into: textEncoder)
        let tok = try Qwen3Tokenizer.textEncoder(baseDir: Self.modelBaseDir)

        // Realistic multi-stanza lyrics — 55+ tokens — that previously triggered overflow.
        let lyrics = """
        # Languages
        en

        # Lyric
        [verse]
        Sunlight through the window pane
        Coffee steam and soft refrain
        Pages turn without a sound
        Peace is what I finally found

        [chorus]
        Drifting slow through golden haze
        Lost inside these quiet days<|endoftext|>
        """
        let ids = tok.encode(lyrics)
        XCTAssertGreaterThan(ids.count, 30, "Need >30 tokens to exercise the overflow regime")

        let inputIds = MLXArray(ids.map { Int32($0) }).reshaped([1, ids.count])
        let lyricEmbeds = textEncoder.embed(inputIds)
        let encoded = dit.lyricEncoder(lyricEmbeds)
        eval(encoded)

        let sumValue = encoded.sum().item(Float.self)
        XCTAssertTrue(sumValue.isFinite, "lyric_encoder must produce finite output (was NaN)")
        let absMax = abs(encoded).max().item(Float.self)
        XCTAssertLessThan(absMax, 100,
                          "lyric_encoder output magnitude should be bounded after the final RMSNorm; saw \(absMax)")
    }

    /// Proves the lyric+text path of `NativeInferenceEngine.buildEncoderHiddenStates`
    /// runs on real weights and produces a finite, well-shaped tensor.
    func testEndToEndConditioningPipeline() throws {
        try skipIfMissing()
        let p = Self.modelBaseDir.appendingPathComponent("dit/dit_weights.safetensors")
        guard FileManager.default.fileExists(atPath: p.path) else {
            throw XCTSkip("DiT weights missing — run tools/convert_weights.py")
        }

        let dit = ACEStepDiT()
        try DiTWeightLoader.load(baseDir: Self.modelBaseDir, into: dit)

        let textEncoder = Qwen3EncoderModel()
        try Qwen3EncoderWeightLoader.load(baseDir: Self.modelBaseDir, into: textEncoder)

        let tok = try Qwen3Tokenizer.textEncoder(baseDir: Self.modelBaseDir)

        // Tokenize and embed a short lyric.
        let lyric = "# Languages\nen\n\n# Lyric\nLa la la<|endoftext|>"
        let lyricIds = tok.encode(lyric)
        let lyricInput = MLXArray(lyricIds.map { Int32($0) }).reshaped([1, lyricIds.count])
        let lyricEmbeds = textEncoder.embed(lyricInput)             // [1, S_lyric, 1024]
        let lyricEncoded = dit.lyricEncoder(lyricEmbeds)            // [1, S_lyric, 2048]
        eval(lyricEncoded)
        XCTAssertEqual(lyricEncoded.shape, [1, lyricIds.count, 2048])

        // Tokenize and run text full forward.
        let textTokens = tok.encode("# Caption\nlofi beats<|endoftext|>")
        let textInput = MLXArray(textTokens.map { Int32($0) }).reshaped([1, textTokens.count])
        let textHidden = textEncoder.encode(textInput)              // [1, S_text, 1024]
        let textProjected = dit.textProjector(textHidden)           // [1, S_text, 2048]
        eval(textProjected)
        XCTAssertEqual(textProjected.shape, [1, textTokens.count, 2048])

        // Pack
        let lyricMask = MLXArray.ones([1, lyricIds.count]).asType(.int32)
        let textMask  = MLXArray.ones([1, textTokens.count]).asType(.int32)
        let (packed, packedMask) = PackSequences.pack(lyricEncoded, textProjected, lyricMask, textMask)
        eval(packed); eval(packedMask)

        XCTAssertEqual(packed.shape, [1, lyricIds.count + textTokens.count, 2048])
        XCTAssertTrue(packed.sum().item(Float.self).isFinite,
                      "Packed cross-attention conditioning must be finite")

        let absMean = abs(packed).mean().item(Float.self)
        XCTAssertGreaterThan(absMean, 1e-4,
                             "Packed conditioning should be non-degenerate")
    }
}

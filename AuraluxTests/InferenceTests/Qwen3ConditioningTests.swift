import XCTest
import MLX
import MLXNN
import MLXRandom
@testable import Auralux

/// Tests for the Qwen3-Embedding text/lyric conditioning pipeline.
///
/// Goldens are produced by:
///   * `transformers.AutoTokenizer.from_pretrained("ACE-Step/Ace-Step1.5/Qwen3-Embedding-0.6B")`
///     for tokenizer values, and
///   * upstream `pack_sequences` from `modeling_acestep_v15_turbo.py:141-172`
///     for packing.
///
/// Tokenizer / weights tests are skipped automatically when converted weights are
/// not present locally (they require a 1.2 GB download).
final class Qwen3ConditioningTests: XCTestCase {

    // MARK: - PackSequences

    /// Case 1: both batches fully valid — packed is just concatenation.
    func testPackSequencesAllValid() {
        let h1 = MLXArray([1, 2, 3, 4, 5, 6] as [Float]).reshaped([1, 3, 2])
        let h2 = MLXArray([10, 20, 30, 40] as [Float]).reshaped([1, 2, 2])
        let m1 = MLXArray([1, 1, 1] as [Int32]).reshaped([1, 3])
        let m2 = MLXArray([1, 1] as [Int32]).reshaped([1, 2])

        let (packed, mask) = PackSequences.pack(h1, h2, m1, m2)
        eval(packed); eval(mask)

        XCTAssertEqual(packed.shape, [1, 5, 2])
        XCTAssertEqual(mask.shape,   [1, 5])
        XCTAssertEqual(packed.flattened().asArray(Float.self),
                       [1, 2, 3, 4, 5, 6, 10, 20, 30, 40])
        XCTAssertEqual(mask.flattened().asArray(Int32.self),
                       [1, 1, 1, 1, 1])
    }

    /// Case 2: mixed validity — valid tokens move to front, padded to back.
    /// Upstream golden:
    ///   h1 = [[1,2],[3,4],[5,6]]  m1 = [1,0,1]
    ///   h2 = [[10,20],[30,40]]    m2 = [1,0]
    ///   packed = [[1,2],[5,6],[10,20],[3,4],[30,40]]   mask = [1,1,1,0,0]
    func testPackSequencesMixedValidity() {
        let h1 = MLXArray([1, 2, 3, 4, 5, 6] as [Float]).reshaped([1, 3, 2])
        let h2 = MLXArray([10, 20, 30, 40] as [Float]).reshaped([1, 2, 2])
        let m1 = MLXArray([1, 0, 1] as [Int32]).reshaped([1, 3])
        let m2 = MLXArray([1, 0] as [Int32]).reshaped([1, 2])

        let (packed, mask) = PackSequences.pack(h1, h2, m1, m2)
        eval(packed); eval(mask)

        XCTAssertEqual(packed.shape, [1, 5, 2])

        let valid = packed[0..., ..<3, 0...]   // Only the first 3 positions are valid
        eval(valid)
        let validValues = Set(valid.flattened().asArray(Float.self))
        // The valid positions must be exactly {h1[0], h1[2], h2[0]} = {(1,2),(5,6),(10,20)}.
        // We compare as a set because argsort stability across rows isn't guaranteed,
        // but the *set* of valid tokens is invariant.
        XCTAssertEqual(validValues, [1, 2, 5, 6, 10, 20])

        XCTAssertEqual(mask.flattened().asArray(Int32.self),
                       [1, 1, 1, 0, 0])
    }

    func testPackSequencesBatched() {
        // Upstream golden (batched):
        // h1 = [[[1,2],[3,4]],[[5,6],[7,8]]]  m1 = [[1,0],[1,1]]
        // h2 = [[[10,20]],[[30,40]]]          m2 = [[1],[0]]
        // packed = [[[1,2],[10,20],[3,4]], [[5,6],[7,8],[30,40]]]
        // mask   = [[1,1,0],[1,1,0]]
        let h1 = MLXArray([1, 2, 3, 4, 5, 6, 7, 8] as [Float]).reshaped([2, 2, 2])
        let h2 = MLXArray([10, 20, 30, 40] as [Float]).reshaped([2, 1, 2])
        let m1 = MLXArray([1, 0, 1, 1] as [Int32]).reshaped([2, 2])
        let m2 = MLXArray([1, 0] as [Int32]).reshaped([2, 1])

        let (packed, mask) = PackSequences.pack(h1, h2, m1, m2)
        eval(packed); eval(mask)

        XCTAssertEqual(packed.shape, [2, 3, 2])
        XCTAssertEqual(mask.flattened().asArray(Int32.self),
                       [1, 1, 0, 1, 1, 0])
    }

    // MARK: - Qwen3 encoder shape

    func testQwen3EncoderEmbedShape() {
        // Mini config — keep tiny so weight init is fast.
        let cfg = Qwen3EncoderConfig(
            vocabSize: 1000, hiddenSize: 32,
            numHiddenLayers: 2, numAttentionHeads: 4,
            numKeyValueHeads: 2, headDim: 8,
            intermediateSize: 64
        )
        let m = Qwen3EncoderModel(config: cfg)
        let ids = MLXArray([1, 2, 3, 4] as [Int32]).reshaped([1, 4])
        let emb = m.embed(ids)
        eval(emb)
        XCTAssertEqual(emb.shape, [1, 4, cfg.hiddenSize])
        XCTAssertTrue(emb.sum().item(Float.self).isFinite)
    }

    func testQwen3EncoderEncodeShape() {
        let cfg = Qwen3EncoderConfig(
            vocabSize: 1000, hiddenSize: 32,
            numHiddenLayers: 2, numAttentionHeads: 4,
            numKeyValueHeads: 2, headDim: 8,
            intermediateSize: 64
        )
        let m = Qwen3EncoderModel(config: cfg)
        let ids = MLXArray([1, 2, 3, 4, 5, 6] as [Int32]).reshaped([1, 6])
        let h = m.encode(ids)
        eval(h)
        XCTAssertEqual(h.shape, [1, 6, cfg.hiddenSize])
        XCTAssertTrue(h.sum().item(Float.self).isFinite)
    }

    // MARK: - Tokenizer goldens (require converted weights)

    /// Goldens are produced by `tokenizer.encode(text, add_special_tokens=False)`.
    /// We compare against `encodeRaw` since Swift's `encode` appends the trailing
    /// pad/EOS token (matching `tokenizer(text)` __call__ default).
    func testQwen3TokenizerHelloGoldenRaw() throws {
        try skipIfTextWeightsMissing()
        let tok = try Qwen3Tokenizer.textEncoder(baseDir: Self.modelBaseDir)
        XCTAssertEqual(tok.encodeRaw("Hello"), [9707])
    }

    func testQwen3TokenizerEndoftextSpecialGoldenRaw() throws {
        try skipIfTextWeightsMissing()
        let tok = try Qwen3Tokenizer.textEncoder(baseDir: Self.modelBaseDir)
        // <|endoftext|> must resolve to a single special token (id 151643), not raw bytes.
        XCTAssertEqual(tok.encodeRaw("<|endoftext|>"), [151643])
    }

    func testQwen3TokenizerLyricFormatGoldenRaw() throws {
        try skipIfTextWeightsMissing()
        let tok = try Qwen3Tokenizer.textEncoder(baseDir: Self.modelBaseDir)
        let s = "# Languages\nen\n\n# Lyric\nHello<|endoftext|>"
        XCTAssertEqual(
            tok.encodeRaw(s),
            [2, 54964, 198, 268, 271, 2, 15953, 2216, 198, 9707, 151643]
        )
    }

    func testQwen3TokenizerLaLaLaGoldenRaw() throws {
        try skipIfTextWeightsMissing()
        let tok = try Qwen3Tokenizer.textEncoder(baseDir: Self.modelBaseDir)
        let s = "# Languages\nen\n\n# Lyric\nLa la la<|endoftext|>"
        XCTAssertEqual(
            tok.encodeRaw(s),
            [2, 54964, 198, 268, 271, 2, 15953, 2216, 198, 8747, 1187, 1187, 151643]
        )
    }

    /// `encode` (the default path used by NativeInferenceEngine) appends one extra
    /// 151643 at the end to mirror upstream's `tokenizer(text)` __call__.
    func testQwen3TokenizerAppendsTrailingPadToken() throws {
        try skipIfTextWeightsMissing()
        let tok = try Qwen3Tokenizer.textEncoder(baseDir: Self.modelBaseDir)
        XCTAssertEqual(tok.encode("Hello"), [9707, 151643])
        let s = "# Languages\nen\n\n# Lyric\nHello<|endoftext|>"
        XCTAssertEqual(
            tok.encode(s),
            [2, 54964, 198, 268, 271, 2, 15953, 2216, 198, 9707, 151643, 151643]
        )
    }

    // MARK: - Helpers

    private static let modelBaseDir: URL = FileUtilities.modelDirectory
        .appendingPathComponent("ace-step-v1.5-mlx", isDirectory: true)

    private func skipIfTextWeightsMissing() throws {
        let path = Self.modelBaseDir.appendingPathComponent("text/text_vocab.json")
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw XCTSkip("text encoder weights not present at \(path.path) — run tools/convert_weights.py")
        }
    }
}

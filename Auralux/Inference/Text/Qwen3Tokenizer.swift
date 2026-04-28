import Foundation

/// Tokenizer for Qwen3-Embedding-0.6B, used for both lyric and text conditioning.
///
/// Wraps the existing byte-level BPE (`BPETokenizer`) and adds upstream-correct
/// handling of *added* / special tokens (e.g. `<|endoftext|>` → 151643).
/// The upstream `_format_lyrics` produces strings like
/// `"# Languages\nen\n\n# Lyric\nHello<|endoftext|>"`, where `<|endoftext|>` must
/// resolve to a single special-token id rather than ~14 raw byte tokens.
struct Qwen3Tokenizer {

    private let bpe: BPETokenizer
    /// Map of special-token literal (e.g. `"<|endoftext|>"`) → id.
    /// Loaded from `added_tokens.json`. Sorted by length desc to prefer long matches.
    private let specialTokens: [(token: String, id: Int)]
    /// Token id appended at the end of `encode(...)` to mirror upstream's
    /// `tokenizer(...)` __call__ default (`add_special_tokens=True` adds the
    /// pad/eos token at the end). For Qwen3-Embedding-0.6B this is 151643
    /// (`<|endoftext|>` / pad_token_id).
    private let appendedSpecialToken: Int?

    enum Source: Sendable {
        case textEncoder    // Qwen3-Embedding-0.6B
        case lm             // 5Hz LM
    }

    // MARK: - Init

    init(vocabURL: URL, mergesURL: URL, addedTokensURL: URL?, appendedSpecialToken: Int? = 151643) throws {
        bpe = try BPETokenizer(vocabURL: vocabURL, mergesURL: mergesURL)

        var added: [(String, Int)] = []
        if let addedTokensURL,
           FileManager.default.fileExists(atPath: addedTokensURL.path) {
            let data = try Data(contentsOf: addedTokensURL)
            let dict = try JSONDecoder().decode([String: Int].self, from: data)
            added = dict.map { ($0.key, $0.value) }
        }
        // Longest-match-first to prevent prefix-of-special-token collisions.
        added.sort { $0.0.count > $1.0.count }
        specialTokens = added
        self.appendedSpecialToken = appendedSpecialToken
    }

    /// Convenience initializer for the Qwen3-Embedding text encoder living in
    /// `<baseDir>/text/text_*.json` / `text_*.txt`.
    static func textEncoder(baseDir: URL) throws -> Qwen3Tokenizer {
        let dir = baseDir.appendingPathComponent("text")
        return try Qwen3Tokenizer(
            vocabURL:        dir.appendingPathComponent("text_vocab.json"),
            mergesURL:       dir.appendingPathComponent("text_merges.txt"),
            addedTokensURL:  dir.appendingPathComponent("text_added_tokens.json")
        )
    }

    // MARK: - Encoding

    /// Encode a string into token IDs, honoring registered special tokens.
    ///
    /// Algorithm: scan left-to-right. At each position, attempt the longest
    /// matching special-token literal (longest-first); on hit, emit the special
    /// id and advance. Otherwise, accumulate raw text and pass through byte-level
    /// BPE when a special token is hit (or at end of input). Finally, append
    /// `appendedSpecialToken` (default: pad/EOS 151643) to mirror upstream's
    /// `tokenizer(text)` default behavior.
    func encode(_ text: String) -> [Int] {
        var ids = encodeRaw(text)
        if let extra = appendedSpecialToken {
            ids.append(extra)
        }
        return ids
    }

    /// Encode without the trailing special token (matches HF's `add_special_tokens=False`).
    /// Useful for tests that need bit-exact comparison with `tokenizer.encode(text, add_special_tokens=False)`.
    func encodeRaw(_ text: String) -> [Int] {
        guard !text.isEmpty else { return [] }
        if specialTokens.isEmpty {
            return bpe.encode(text)
        }

        var ids: [Int] = []
        var buffer = ""

        let scalars = Array(text)
        var i = 0
        while i < scalars.count {
            var matched: (String, Int)? = nil
            for (tok, tid) in specialTokens {
                if matchesAt(scalars: scalars, index: i, prefix: tok) {
                    matched = (tok, tid)
                    break
                }
            }
            if let (tok, tid) = matched {
                if !buffer.isEmpty {
                    ids.append(contentsOf: bpe.encode(buffer))
                    buffer.removeAll(keepingCapacity: true)
                }
                ids.append(tid)
                i += tok.count
            } else {
                buffer.append(scalars[i])
                i += 1
            }
        }
        if !buffer.isEmpty {
            ids.append(contentsOf: bpe.encode(buffer))
        }
        return ids
    }

    private func matchesAt(scalars: [Character], index: Int, prefix: String) -> Bool {
        let prefixChars = Array(prefix)
        if scalars.count - index < prefixChars.count { return false }
        for k in 0..<prefixChars.count {
            if scalars[index + k] != prefixChars[k] { return false }
        }
        return true
    }
}

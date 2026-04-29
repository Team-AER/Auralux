import Foundation

/// Qwen2 / GPT-2 style byte-level BPE tokenizer.
/// Loads vocabulary from `lm_vocab.json` and merge rules from `lm_merges.txt`.
struct BPETokenizer {

    private let vocab: [String: Int]
    private let mergeRanks: [String: Int]   // "tok1 tok2" → priority (lower = higher)

    // Maps each raw byte (0-255) to its single-character byte-level representation.
    // Printable bytes map to themselves; others map to consecutive Unicode chars from U+0100.
    private static let byteEncoder: [UInt8: String] = {
        var result = [UInt8: String]()
        var extraN: UInt32 = 0
        for byte in 0..<256 {
            let b = UInt8(byte)
            let isPrintable = (33...126).contains(byte) || (161...172).contains(byte) || (174...255).contains(byte)
            let codePoint: UInt32
            if isPrintable {
                codePoint = UInt32(byte)
            } else {
                codePoint = 256 + extraN
                extraN += 1
            }
            result[b] = String(Character(Unicode.Scalar(codePoint)!))
        }
        return result
    }()

    // MARK: - Init

    init(vocabURL: URL, mergesURL: URL) throws {
        let vocabData = try Data(contentsOf: vocabURL)
        vocab = try JSONDecoder().decode([String: Int].self, from: vocabData)

        let mergesText = try String(contentsOf: mergesURL, encoding: .utf8)
        var ranks = [String: Int]()
        ranks.reserveCapacity(160_000)
        var rank = 0
        for line in mergesText.split(separator: "\n", omittingEmptySubsequences: true) {
            let s = String(line)
            if s.hasPrefix("#") { continue }
            ranks[s] = rank
            rank += 1
        }
        mergeRanks = ranks
    }

    // MARK: - Public

    /// Encodes `text` into BPE token IDs.
    func encode(_ text: String) -> [Int] {
        guard !text.isEmpty else { return [] }

        // Pre-tokenise: split on spaces; each space moves to the front of the next word.
        // Bytes are then byte-encoded per word so space (0x20) → 'Ġ' (U+0120).
        var allIds: [Int] = []
        let encoder = Self.byteEncoder

        var groupStart = text.utf8.startIndex
        var i = text.utf8.startIndex

        func flush(to end: String.UTF8View.Index, includeLeadingSpace: Bool) {
            let slice = text.utf8[groupStart..<end]
            if includeLeadingSpace {
                var syms = [encoder[0x20]!]   // Ġ
                syms.append(contentsOf: slice.map { encoder[$0] ?? "?" })
                allIds.append(contentsOf: bpeSymbols(syms))
            } else {
                let syms = slice.map { encoder[$0] ?? "?" }
                if !syms.isEmpty { allIds.append(contentsOf: bpeSymbols(syms)) }
            }
        }

        var foundFirst = false
        var pendingSpace = false

        while i < text.utf8.endIndex {
            let byte = text.utf8[i]
            if byte == 0x20 {
                if i > groupStart {
                    flush(to: i, includeLeadingSpace: pendingSpace && foundFirst)
                    foundFirst = true
                }
                groupStart = text.utf8.index(after: i)
                pendingSpace = true
            }
            i = text.utf8.index(after: i)
        }
        // Last group
        if groupStart < text.utf8.endIndex {
            flush(to: text.utf8.endIndex, includeLeadingSpace: pendingSpace && foundFirst)
        } else if groupStart == text.utf8.endIndex && !foundFirst {
            // empty result
        }

        return allIds
    }

    // MARK: - Private

    private func bpeSymbols(_ symbols: [String]) -> [Int] {
        var syms = symbols
        guard syms.count > 1 else {
            return syms.map { vocab[$0] ?? 0 }
        }

        while syms.count > 1 {
            var bestRank = Int.max
            var bestI = -1
            for i in 0..<syms.count - 1 {
                let key = syms[i] + " " + syms[i + 1]
                if let rank = mergeRanks[key], rank < bestRank {
                    bestRank = rank
                    bestI = i
                }
            }
            guard bestI >= 0 else { break }

            let first = syms[bestI], second = syms[bestI + 1]
            var next: [String] = []
            next.reserveCapacity(syms.count - 1)
            var j = 0
            while j < syms.count {
                if j + 1 < syms.count && syms[j] == first && syms[j + 1] == second {
                    next.append(first + second)
                    j += 2
                } else {
                    next.append(syms[j])
                    j += 1
                }
            }
            syms = next
        }

        return syms.map { vocab[$0] ?? 0 }
    }
}

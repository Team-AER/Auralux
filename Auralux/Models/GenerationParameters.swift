import Foundation

struct GenerationParameters: Codable, Hashable, Sendable {
    var prompt: String
    var lyrics: String
    var tags: [String]
    var duration: TimeInterval
    var variance: Double
    var seed: Int?
    /// ISO-639-1 / upstream language code for the lyric language header.
    /// Default `"en"`. Use `"unknown"` if you don't want to bias the model.
    /// See `acestep/constants.py` SUPPORTED_LANGUAGES — 51 codes including
    /// `en zh ja ko es fr de it pt ru ar hi bn ...`.
    var language: String

    static let `default` = GenerationParameters(
        prompt: "chill lofi piano",
        lyrics: """
            [verse]
            Sunlight through the window pane
            Coffee steam and soft refrain
            Pages turn without a sound
            Peace is what I finally found

            [chorus]
            Drifting slow through golden haze
            Lost inside these quiet days
            """,
        tags: ["lofi", "piano", "chill"],
        duration: 30,
        variance: 0.5,
        seed: nil,
        language: "en"
    )

    enum CodingKeys: String, CodingKey {
        case prompt, lyrics, tags, duration, variance, seed, language
    }

    init(
        prompt: String,
        lyrics: String,
        tags: [String],
        duration: TimeInterval,
        variance: Double,
        seed: Int?,
        language: String = "en"
    ) {
        self.prompt   = prompt
        self.lyrics   = lyrics
        self.tags     = tags
        self.duration = duration
        self.variance = variance
        self.seed     = seed
        self.language = language
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        prompt   = try c.decode(String.self,       forKey: .prompt)
        lyrics   = try c.decode(String.self,       forKey: .lyrics)
        tags     = try c.decode([String].self,     forKey: .tags)
        duration = try c.decode(TimeInterval.self, forKey: .duration)
        variance = try c.decode(Double.self,       forKey: .variance)
        seed     = try c.decodeIfPresent(Int.self, forKey: .seed)
        language = try c.decodeIfPresent(String.self, forKey: .language) ?? "en"
    }
}

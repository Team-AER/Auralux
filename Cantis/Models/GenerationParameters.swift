import Foundation

/// A second-range to repaint, in seconds from the start of the clip.
/// Inclusive on both ends. `ClosedRange` isn't `Codable` so we use this.
struct RepaintRange: Codable, Hashable, Sendable {
    var start: Double
    var end: Double
}

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

    // ── DiT knobs ────────────────────────────────────────────────────────────
    var mode: GenerationMode
    /// Number of denoising steps. Upstream clamps to 1...20.
    var numSteps: Int
    /// Schedule shift in {1.0, 2.0, 3.0}. Higher compresses late-denoising steps.
    var scheduleShift: Double
    /// Classifier-free guidance scale. Ignored on turbo (CFG is distilled into
    /// the weights); applied on base/sft as a twin unconditional pass.
    var cfgScale: Double

    // ── Mode-specific inputs (only set for the matching mode) ───────────────
    var referAudioURL: URL?
    var sourceAudioURL: URL?
    var repaintMaskRanges: [RepaintRange]
    var repaintCrossfadeFrames: Int
    var repaintInjectionRatio: Double

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
        case mode, numSteps, scheduleShift, cfgScale
        case referAudioURL, sourceAudioURL
        case repaintMaskRanges, repaintCrossfadeFrames, repaintInjectionRatio
    }

    init(
        prompt: String,
        lyrics: String,
        tags: [String],
        duration: TimeInterval,
        variance: Double,
        seed: Int?,
        language: String = "en",
        mode: GenerationMode = .text2music,
        numSteps: Int = 8,
        scheduleShift: Double = 1.0,
        cfgScale: Double = 1.0,
        referAudioURL: URL? = nil,
        sourceAudioURL: URL? = nil,
        repaintMaskRanges: [RepaintRange] = [],
        repaintCrossfadeFrames: Int = 10,
        repaintInjectionRatio: Double = 0.5
    ) {
        self.prompt   = prompt
        self.lyrics   = lyrics
        self.tags     = tags
        self.duration = duration
        self.variance = variance
        self.seed     = seed
        self.language = language
        self.mode = mode
        self.numSteps = numSteps
        self.scheduleShift = scheduleShift
        self.cfgScale = cfgScale
        self.referAudioURL = referAudioURL
        self.sourceAudioURL = sourceAudioURL
        self.repaintMaskRanges = repaintMaskRanges
        self.repaintCrossfadeFrames = repaintCrossfadeFrames
        self.repaintInjectionRatio = repaintInjectionRatio
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
        // All new fields default-decode so existing presets / persisted history
        // entries keep working after the schema change.
        mode = try c.decodeIfPresent(GenerationMode.self, forKey: .mode) ?? .text2music
        numSteps = try c.decodeIfPresent(Int.self, forKey: .numSteps) ?? 8
        scheduleShift = try c.decodeIfPresent(Double.self, forKey: .scheduleShift) ?? 1.0
        cfgScale = try c.decodeIfPresent(Double.self, forKey: .cfgScale) ?? 1.0
        referAudioURL = try c.decodeIfPresent(URL.self, forKey: .referAudioURL)
        sourceAudioURL = try c.decodeIfPresent(URL.self, forKey: .sourceAudioURL)
        repaintMaskRanges = try c.decodeIfPresent([RepaintRange].self,
                                                  forKey: .repaintMaskRanges) ?? []
        repaintCrossfadeFrames = try c.decodeIfPresent(Int.self,
                                                       forKey: .repaintCrossfadeFrames) ?? 10
        repaintInjectionRatio = try c.decodeIfPresent(Double.self,
                                                      forKey: .repaintInjectionRatio) ?? 0.5
    }
}

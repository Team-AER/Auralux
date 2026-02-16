import Foundation

struct GenerationParameters: Codable, Hashable, Sendable {
    var prompt: String
    var lyrics: String
    var tags: [String]
    var duration: TimeInterval
    var variance: Double
    var seed: Int?

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
        seed: nil
    )
}

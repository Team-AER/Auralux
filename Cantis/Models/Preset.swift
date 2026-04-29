import Foundation
import SwiftData

@Model
final class Preset {
    @Attribute(.unique) var id: UUID
    var name: String
    var summary: String
    var prompt: String
    var lyricTemplate: String
    var tags: [String]
    var duration: TimeInterval
    var variance: Double
    var createdAt: Date
    var updatedAt: Date

    init(
        id: UUID = UUID(),
        name: String,
        summary: String,
        prompt: String,
        lyricTemplate: String = "",
        tags: [String] = [],
        duration: TimeInterval = 30,
        variance: Double = 0.5,
        createdAt: Date = .now,
        updatedAt: Date = .now
    ) {
        self.id = id
        self.name = name
        self.summary = summary
        self.prompt = prompt
        self.lyricTemplate = lyricTemplate
        self.tags = tags
        self.duration = duration
        self.variance = variance
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }

    var parameters: GenerationParameters {
        GenerationParameters(
            prompt: prompt,
            lyrics: lyricTemplate,
            tags: tags,
            duration: duration,
            variance: variance,
            seed: nil
        )
    }
}

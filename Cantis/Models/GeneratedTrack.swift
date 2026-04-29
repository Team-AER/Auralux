import Foundation
import SwiftData

@Model
final class GeneratedTrack {
    @Attribute(.unique) var id: UUID
    var title: String
    var prompt: String
    var lyrics: String
    var tags: [String]
    var duration: TimeInterval
    var variance: Double
    var seed: Int?
    // Unique per-job identifier from the inference server; prevents duplicate
    // persists if the poll loop fires more than once on the completed status.
    @Attribute(.unique) var generationID: String
    var audioFilePath: String?
    var format: String
    var isFavorite: Bool
    var createdAt: Date

    init(
        id: UUID = UUID(),
        title: String,
        prompt: String,
        lyrics: String,
        tags: [String],
        duration: TimeInterval,
        variance: Double,
        seed: Int?,
        generationID: String,
        audioFilePath: String? = nil,
        format: String = "wav",
        isFavorite: Bool = false,
        createdAt: Date = .now
    ) {
        self.id = id
        self.title = title
        self.prompt = prompt
        self.lyrics = lyrics
        self.tags = tags
        self.duration = duration
        self.variance = variance
        self.seed = seed
        self.generationID = generationID
        self.audioFilePath = audioFilePath
        self.format = format
        self.isFavorite = isFavorite
        self.createdAt = createdAt
    }
}

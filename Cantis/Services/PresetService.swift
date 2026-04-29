import Foundation
import SwiftData

@MainActor
final class PresetService {
    private let context: ModelContext

    init(context: ModelContext) {
        self.context = context
    }

    func bootstrapFromBundleIfNeeded() throws {
        let existing = try context.fetchCount(FetchDescriptor<Preset>())
        guard existing == 0 else { return }

        guard let url = Bundle.module.url(forResource: "starter_presets", withExtension: "json", subdirectory: "Presets") else {
            return
        }

        let data = try Data(contentsOf: url)
        let presets = try JSONDecoder().decode([PresetSeed].self, from: data)

        for seed in presets {
            context.insert(
                Preset(
                    name: seed.name,
                    summary: seed.summary,
                    prompt: seed.prompt,
                    lyricTemplate: seed.lyricTemplate,
                    tags: seed.tags,
                    duration: seed.duration,
                    variance: seed.variance
                )
            )
        }
        try context.save()
    }

    func fetchAll() throws -> [Preset] {
        try context.fetch(
            FetchDescriptor<Preset>(
                sortBy: [SortDescriptor(\Preset.updatedAt, order: .reverse)]
            )
        )
    }

    func save(_ preset: Preset) throws {
        preset.updatedAt = .now
        if preset.modelContext == nil {
            context.insert(preset)
        }
        try context.save()
    }

    func delete(_ preset: Preset) throws {
        context.delete(preset)
        try context.save()
    }
}

private struct PresetSeed: Codable {
    var name: String
    var summary: String
    var prompt: String
    var lyricTemplate: String
    var tags: [String]
    var duration: TimeInterval
    var variance: Double
}

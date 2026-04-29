import SwiftData
import SwiftUI

struct PresetListView: View {
    @Environment(GenerationViewModel.self) private var generationViewModel
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \Preset.updatedAt, order: .reverse) private var presets: [Preset]

    var body: some View {
        if presets.isEmpty {
            Text("No presets")
                .foregroundStyle(.secondary)
        } else {
            ForEach(presets.prefix(6)) { preset in
                Button {
                    generationViewModel.applyPreset(preset)
                } label: {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(preset.name)
                        Text(preset.summary)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }
                .buttonStyle(.plain)
            }
        }
    }
}

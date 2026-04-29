import SwiftUI

struct HistoryItemView: View {
    let track: GeneratedTrack

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(track.title)
                        .font(.headline)
                    if track.isFavorite {
                        Image(systemName: "star.fill")
                            .foregroundStyle(.yellow)
                    }
                }
                Text(track.prompt)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                Text(track.createdAt.formatted(date: .abbreviated, time: .shortened))
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
            Spacer()
            Text("\(Int(track.duration))s")
                .font(.caption)
                .padding(6)
                .background(.thinMaterial, in: Capsule())
        }
        .padding(.vertical, 4)
    }
}

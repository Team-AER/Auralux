import SwiftData
import SwiftUI

struct RecentListView: View {
    @Environment(HistoryViewModel.self) private var historyViewModel
    @Query(sort: \GeneratedTrack.createdAt, order: .reverse) private var tracks: [GeneratedTrack]

    var body: some View {
        if tracks.isEmpty {
            Text("No history")
                .foregroundStyle(.secondary)
        } else {
            ForEach(tracks.prefix(6)) { track in
                Button {
                    historyViewModel.selectedTrack = track
                } label: {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(track.title)
                            .lineLimit(1)
                        Text(track.createdAt.formatted(date: .abbreviated, time: .shortened))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .buttonStyle(.plain)
            }
        }
    }
}

import Foundation
import Observation
import SwiftData

@MainActor
@Observable
final class HistoryViewModel {
    var query = ""
    var tracks: [GeneratedTrack] = []
    var selectedTrack: GeneratedTrack?
    var errorMessage: String?

    func refresh(context: ModelContext) {
        let service = HistoryService(context: context)
        do {
            if query.isEmpty {
                tracks = try service.recent(limit: 100)
            } else {
                tracks = try service.search(query: query)
            }
            if let selectedTrack {
                self.selectedTrack = tracks.first(where: { $0.id == selectedTrack.id })
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func toggleFavorite(_ track: GeneratedTrack, context: ModelContext) {
        let service = HistoryService(context: context)
        do {
            try service.setFavorite(track, isFavorite: !track.isFavorite)
            refresh(context: context)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func delete(_ track: GeneratedTrack, context: ModelContext) {
        let service = HistoryService(context: context)
        if selectedTrack?.id == track.id { selectedTrack = nil }
        do {
            try service.delete(track)
            refresh(context: context)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func deleteAll(context: ModelContext) {
        let service = HistoryService(context: context)
        selectedTrack = nil
        do {
            try service.deleteAll()
            refresh(context: context)
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}

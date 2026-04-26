import Foundation
import SwiftData

@MainActor
final class HistoryService {
    private let context: ModelContext

    init(context: ModelContext) {
        self.context = context
    }

    func insert(_ track: GeneratedTrack) throws {
        context.insert(track)
        try context.save()
    }

    func recent(limit: Int = 25) throws -> [GeneratedTrack] {
        var descriptor = FetchDescriptor<GeneratedTrack>(
            sortBy: [SortDescriptor(\GeneratedTrack.createdAt, order: .reverse)]
        )
        descriptor.fetchLimit = limit
        return try context.fetch(descriptor)
    }

    func search(query: String) throws -> [GeneratedTrack] {
        let descriptor = FetchDescriptor<GeneratedTrack>(
            predicate: #Predicate { track in
                track.title.localizedStandardContains(query)
                || track.prompt.localizedStandardContains(query)
            },
            sortBy: [SortDescriptor(\GeneratedTrack.createdAt, order: .reverse)]
        )
        return try context.fetch(descriptor)
    }

    func setFavorite(_ track: GeneratedTrack, isFavorite: Bool) throws {
        track.isFavorite = isFavorite
        try context.save()
    }

    func delete(_ track: GeneratedTrack) throws {
        if let path = track.audioFilePath,
           let url = FileUtilities.resolveAudioPath(path) {
            try? FileManager.default.removeItem(at: url)
        }
        context.delete(track)
        try context.save()
    }

    /// Removes audio files in the Generated directory that are not referenced
    /// by any GeneratedTrack row. Safe to call on every launch.
    /// The SwiftData fetch runs on the caller's actor; filesystem I/O is
    /// offloaded to a background task so the main thread is not blocked.
    func reconcileOrphans() async throws {
        let allTracks = try context.fetch(FetchDescriptor<GeneratedTrack>())
        // Use resolvingSymlinksInPath() so that /var/folders and
        // /private/var/folders resolve to the same canonical path.
        let referencedPaths = Set(allTracks.compactMap { $0.audioFilePath }.map {
            FileUtilities.generatedAudioDirectory
                .appendingPathComponent($0)
                .resolvingSymlinksInPath().path
        })

        await Task.detached(priority: .background) {
            let fm = FileManager.default
            let dir = FileUtilities.generatedAudioDirectory
            guard let enumerator = fm.enumerator(
                at: dir, includingPropertiesForKeys: [.isRegularFileKey]
            ) else { return }
            // Use nextObject() — DirectoryEnumerator.makeIterator() is unavailable
            // in async contexts (Swift 6 concurrency restriction on NSEnumerator).
            while let fileURL = enumerator.nextObject() as? URL {
                let isFile = (try? fileURL.resourceValues(
                    forKeys: [.isRegularFileKey]).isRegularFile) == true
                guard isFile else { continue }
                if !referencedPaths.contains(fileURL.resolvingSymlinksInPath().path) {
                    try? fm.removeItem(at: fileURL)
                }
            }
        }.value
    }
}

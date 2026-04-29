import Foundation
import Observation

/// Persists the list of user-added custom models to a JSON file under
/// `~/Library/Application Support/Auralux/CustomModels/registry.json`.
@MainActor
@Observable
final class CustomModelRegistry {
    private(set) var models: [CustomModel] = []

    private let storeURL: URL = FileUtilities.appSupportDirectory
        .appendingPathComponent("CustomModels", isDirectory: true)
        .appendingPathComponent("registry.json")

    init() { load() }

    func model(id: String) -> CustomModel? {
        models.first { $0.id == id }
    }

    func add(_ model: CustomModel) {
        models.removeAll { $0.id == model.id }
        models.append(model)
        save()
    }

    func remove(id: String) {
        models.removeAll { $0.id == id }
        save()
    }

    private func load() {
        guard let data = try? Data(contentsOf: storeURL) else { return }
        models = (try? JSONDecoder().decode([CustomModel].self, from: data)) ?? []
    }

    private func save() {
        let dir = storeURL.deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        guard let data = try? JSONEncoder().encode(models) else { return }
        try? data.write(to: storeURL, options: .atomic)
    }
}

// MARK: - HuggingFace URL parsing

enum HuggingFaceURL {
    /// Parses inputs like `https://huggingface.co/foo/bar`, `huggingface.co/foo/bar`,
    /// or `foo/bar` into a canonical `foo/bar` repoID. Returns nil if invalid.
    static func parseRepoID(_ raw: String) -> String? {
        var s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if s.isEmpty { return nil }
        for prefix in ["https://huggingface.co/", "http://huggingface.co/", "huggingface.co/"] {
            if s.hasPrefix(prefix) {
                s = String(s.dropFirst(prefix.count))
                break
            }
        }
        if s.hasSuffix("/") { s.removeLast() }
        if let q = s.firstIndex(of: "?") { s = String(s[..<q]) }
        if let h = s.firstIndex(of: "#") { s = String(s[..<h]) }
        let parts = s.split(separator: "/", omittingEmptySubsequences: true)
        guard parts.count >= 2 else { return nil }
        let owner = String(parts[0])
        let repo  = String(parts[1])
        guard !owner.isEmpty, !repo.isEmpty else { return nil }
        return "\(owner)/\(repo)"
    }
}

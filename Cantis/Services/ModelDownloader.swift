import Foundation

/// Downloads ACE-Step MLX weights from HuggingFace into the app's model directory.
///
/// Files are downloaded sequentially; overall progress is weighted by file size.
/// Already-present files are skipped so interrupted downloads are resumable.
actor ModelDownloader {

    static let shared = ModelDownloader()

    private static let hfBase = "https://huggingface.co"

    // MARK: - Per-variant manifests

    struct ManifestEntry: Sendable {
        let path: String
        let repoID: String
        let bytes: Int64
    }

    /// Returns the download manifest for a variant. Empty = script-only (XL variants).
    static func manifest(for variant: DiTVariant) -> [ManifestEntry] {
        switch variant {
        case .turbo:
            let repo = "Team-AER/ace-step-v1.5-mlx"
            return [
                ManifestEntry(path: "dit/dit_weights.safetensors",   repoID: repo, bytes: 3_900_000_000),
                ManifestEntry(path: "dit/silence_latent.safetensors", repoID: repo, bytes:     1_920_000),
                ManifestEntry(path: "lm/lm_weights.safetensors",      repoID: repo, bytes: 1_280_000_000),
                ManifestEntry(path: "vae/vae_weights.safetensors",    repoID: repo, bytes:   169_000_000),
                ManifestEntry(path: "text/text_weights.safetensors",  repoID: repo, bytes: 1_200_000_000),
                ManifestEntry(path: "text/text_vocab.json",           repoID: repo, bytes:     5_000_000),
                ManifestEntry(path: "text/text_merges.txt",           repoID: repo, bytes:     1_500_000),
                ManifestEntry(path: "lm/lm_tokenizer.json",          repoID: repo, bytes:    24_300_000),
                ManifestEntry(path: "lm/lm_tokenizer_config.json",   repoID: repo, bytes:    13_100_000),
                ManifestEntry(path: "lm/lm_vocab.json",              repoID: repo, bytes:     2_600_000),
                ManifestEntry(path: "lm/lm_merges.txt",              repoID: repo, bytes:     1_600_000),
                ManifestEntry(path: "lm/lm_added_tokens.json",       repoID: repo, bytes:     2_100_000),
                ManifestEntry(path: "lm/lm_special_tokens_map.json", repoID: repo, bytes:     1_700_000),
                ManifestEntry(path: "lm/lm_chat_template.jinja",     repoID: repo, bytes:         4_100),
                ManifestEntry(path: "lm/lm_config.json",             repoID: repo, bytes:         1_400),
            ]
        case .sft:
            let repo = "Team-AER/ace-step-v1.5-sft-mlx"
            return [
                ManifestEntry(path: "dit/dit_weights.safetensors",   repoID: repo, bytes: 4_790_000_000),
                ManifestEntry(path: "dit/silence_latent.safetensors", repoID: repo, bytes:     1_920_000),
            ]
        case .base:
            let repo = "Team-AER/ace-step-v1.5-base-mlx"
            return [
                ManifestEntry(path: "dit/dit_weights.safetensors",   repoID: repo, bytes: 4_790_000_000),
                ManifestEntry(path: "dit/silence_latent.safetensors", repoID: repo, bytes:     1_920_000),
            ]
        default:
            return []  // XL variants require tools/convert_weights.py
        }
    }

    static func estimatedBytes(for variant: DiTVariant) -> Int64 {
        manifest(for: variant).reduce(0) { $0 + $1.bytes }
    }

    // MARK: - Download

    /// Downloads a variant's weights. For non-turbo variants, also creates symlinks
    /// to the turbo directory so lm/, vae/, and text/ are shared.
    func download(
        variant: DiTVariant,
        to directory: URL,
        turboDirectory: URL,
        onProgress: @escaping @Sendable (Double) -> Void
    ) async throws {
        let entries = Self.manifest(for: variant)
        guard !entries.isEmpty else {
            throw ModelDownloadError.scriptRequired(variant)
        }
        try await performDownload(
            entries:        entries,
            to:             directory,
            needsTurboLinks: variant.requiresTurboBase,
            turboDirectory: turboDirectory,
            onProgress:     onProgress
        )
    }

    /// Downloads a custom ACE-Step-compatible model from an arbitrary HF repo.
    /// File layout matches `baseVariant`'s manifest; lm/vae/text are symlinked
    /// from the turbo directory when `baseVariant != .turbo`.
    func downloadCustom(
        repoID: String,
        baseVariant: DiTVariant,
        to directory: URL,
        turboDirectory: URL,
        onProgress: @escaping @Sendable (Double) -> Void
    ) async throws {
        let proto = Self.manifest(for: baseVariant)
        guard !proto.isEmpty else {
            throw ModelDownloadError.scriptRequired(baseVariant)
        }
        let entries = proto.map { ManifestEntry(path: $0.path, repoID: repoID, bytes: $0.bytes) }
        try await performDownload(
            entries:        entries,
            to:             directory,
            needsTurboLinks: baseVariant.requiresTurboBase,
            turboDirectory: turboDirectory,
            onProgress:     onProgress
        )
    }

    private func performDownload(
        entries: [ManifestEntry],
        to directory: URL,
        needsTurboLinks: Bool,
        turboDirectory: URL,
        onProgress: @escaping @Sendable (Double) -> Void
    ) async throws {
        let totalBytes = entries.reduce(Int64(0)) { $0 + $1.bytes }
        var doneSoFar: Int64 = 0

        for entry in entries {
            let destination = directory.appendingPathComponent(entry.path)

            if FileManager.default.fileExists(atPath: destination.path) {
                doneSoFar += entry.bytes
                onProgress(Double(doneSoFar) / Double(totalBytes))
                continue
            }

            try FileManager.default.createDirectory(
                at: destination.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )

            let urlString = "\(Self.hfBase)/\(entry.repoID)/resolve/main/\(entry.path)"
            guard let remoteURL = URL(string: urlString) else { continue }
            let baseBytes = doneSoFar

            try await downloadFile(from: remoteURL, to: destination) { fileFraction in
                let total = baseBytes + Int64(Double(entry.bytes) * fileFraction)
                onProgress(Double(total) / Double(totalBytes))
            }

            doneSoFar += entry.bytes
        }

        if needsTurboLinks {
            try createCustomSymlinks(in: directory, linkedTo: turboDirectory)
        }

        onProgress(1.0)
    }

    /// Estimated bytes for a custom model (uses the base variant's manifest sizes).
    static func estimatedBytes(forCustomBase variant: DiTVariant) -> Int64 {
        manifest(for: variant).reduce(0) { $0 + $1.bytes }
    }

    /// Shorthand for downloading the turbo variant (used by SetupView).
    func downloadAll(
        to directory: URL,
        onProgress: @escaping @Sendable (Double) -> Void
    ) async throws {
        try await download(variant: .turbo, to: directory, turboDirectory: directory, onProgress: onProgress)
    }

    // MARK: - Private

    /// Symlinks lm/, vae/, text/ from a custom-model directory to the turbo
    /// directory. The two dirs may be siblings (same parent) or arbitrary —
    /// we use absolute paths so external folders work too.
    func createCustomSymlinks(in customDir: URL, linkedTo turboDir: URL) throws {
        let fm = FileManager.default
        for sharedDir in ["lm", "vae", "text"] {
            let link = customDir.appendingPathComponent(sharedDir)
            if fm.fileExists(atPath: link.path) { continue }
            let target = turboDir.appendingPathComponent(sharedDir).path
            try fm.createSymbolicLink(atPath: link.path, withDestinationPath: target)
        }
    }

    private func downloadFile(
        from url: URL,
        to destination: URL,
        onProgress: @escaping @Sendable (Double) -> Void
    ) async throws {
        try await withCheckedThrowingContinuation { continuation in
            let handler = DownloadHandler(
                destination: destination,
                onProgress:  onProgress,
                continuation: continuation
            )
            let session = URLSession(
                configuration: .default,
                delegate: handler,
                delegateQueue: nil
            )
            session.downloadTask(with: url).resume()
        }
    }
}

// MARK: - Errors

enum ModelDownloadError: LocalizedError {
    case scriptRequired(DiTVariant)

    var errorDescription: String? {
        switch self {
        case .scriptRequired(let v):
            return "\(v.displayName) requires script conversion. Run: python tools/convert_weights.py --variant \(v.rawValue)"
        }
    }
}

// MARK: - URLSessionDownloadDelegate bridge

private final class DownloadHandler: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {

    private let destination:  URL
    private let onProgress:   @Sendable (Double) -> Void
    private let continuation: CheckedContinuation<Void, Error>
    private var finished = false

    init(
        destination:  URL,
        onProgress:   @escaping @Sendable (Double) -> Void,
        continuation: CheckedContinuation<Void, Error>
    ) {
        self.destination  = destination
        self.onProgress   = onProgress
        self.continuation = continuation
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData _: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        guard totalBytesExpectedToWrite > 0 else { return }
        onProgress(Double(totalBytesWritten) / Double(totalBytesExpectedToWrite))
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        do {
            if FileManager.default.fileExists(atPath: destination.path) {
                try FileManager.default.removeItem(at: destination)
            }
            try FileManager.default.moveItem(at: location, to: destination)
            complete(with: nil)
        } catch {
            complete(with: error)
        }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        if let error { complete(with: error) }
    }

    private func complete(with error: Error?) {
        guard !finished else { return }
        finished = true
        if let error {
            continuation.resume(throwing: error)
        } else {
            continuation.resume()
        }
    }
}

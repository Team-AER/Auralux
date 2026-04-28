import Foundation

/// Downloads ACE-Step MLX weights from HuggingFace into the app's model directory.
///
/// Files are downloaded sequentially; overall progress is weighted by file size.
/// Already-present files are skipped so interrupted downloads are resumable at
/// the file-granularity level.
actor ModelDownloader {

    static let shared = ModelDownloader()

    private static let baseURL =
        "https://huggingface.co/Team-AER/ace-step-v1.5-mlx/resolve/main/"

    // Paths relative to the model directory root, with approximate byte sizes
    // used for weighted overall-progress calculation.
    private static let manifest: [(path: String, bytes: Int64)] = [
        ("dit/dit_weights.safetensors",   3_900_000_000),
        ("dit/silence_latent.safetensors",    1_920_000),
        ("lm/lm_weights.safetensors",     1_280_000_000),
        ("vae/vae_weights.safetensors",     169_000_000),
        ("lm/lm_tokenizer.json",             24_300_000),
        ("lm/lm_tokenizer_config.json",      13_100_000),
        ("lm/lm_vocab.json",                  2_600_000),
        ("lm/lm_merges.txt",                  1_600_000),
        ("lm/lm_added_tokens.json",           2_100_000),
        ("lm/lm_special_tokens_map.json",     1_700_000),
        ("lm/lm_chat_template.jinja",             4_100),
        ("lm/lm_config.json",                     1_400),
    ]

    private static let totalBytes: Int64 = manifest.reduce(0) { $0 + $1.bytes }

    /// Downloads all missing model files to `directory`.
    ///
    /// - Parameters:
    ///   - directory: The model root (ace-step-v1.5-mlx/). Sub-directories are created automatically.
    ///   - onProgress: Called on an arbitrary thread with an overall fraction in [0, 1].
    func downloadAll(
        to directory: URL,
        onProgress: @escaping @Sendable (Double) -> Void
    ) async throws {
        var doneSoFar: Int64 = 0

        for (path, size) in Self.manifest {
            let destination = directory.appendingPathComponent(path)

            if FileManager.default.fileExists(atPath: destination.path) {
                doneSoFar += size
                onProgress(Double(doneSoFar) / Double(Self.totalBytes))
                continue
            }

            try FileManager.default.createDirectory(
                at: destination.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )

            let remoteURL  = URL(string: Self.baseURL + path)!
            let baseBytes  = doneSoFar

            try await downloadFile(from: remoteURL, to: destination) { fileFraction in
                let total = baseBytes + Int64(Double(size) * fileFraction)
                onProgress(Double(total) / Double(Self.totalBytes))
            }

            doneSoFar += size
        }

        onProgress(1.0)
    }

    // MARK: - Private

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
            // The temp file is deleted after this delegate returns, so move it first.
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

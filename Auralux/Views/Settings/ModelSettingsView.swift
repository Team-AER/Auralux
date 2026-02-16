import SwiftUI

struct ModelSettingsView: View {
    @State private var artifacts: [ModelArtifact] = Self.defaultArtifacts
    @State private var downloadedSet: Set<String> = []
    @State private var downloadingID: String?
    @State private var downloadProgress: Double = 0
    @State private var errorMessage: String?

    private let manager = ModelManagerService()

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Models")
                .font(.headline)

            Text("Models are stored in \(FileUtilities.modelDirectory.path)")
                .font(.caption)
                .foregroundStyle(.secondary)

            if let errorMessage {
                Label(errorMessage, systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.red)
                    .font(.callout)
            }

            VStack(spacing: 0) {
                ForEach(artifacts) { artifact in
                    modelRow(artifact)
                    if artifact.id != artifacts.last?.id {
                        Divider()
                    }
                }
            }
            .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 8))

            HStack {
                Button("Reveal in Finder") {
                    NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: FileUtilities.modelDirectory.path)
                }
                Spacer()
                Button("Refresh") {
                    Task { await refreshStatus() }
                }
            }
            .font(.callout)
        }
        .task {
            await refreshStatus()
        }
    }

    @ViewBuilder
    private func modelRow(_ artifact: ModelArtifact) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(artifact.name)
                    .font(.body.weight(.medium))
                Text(Self.formatBytes(artifact.sizeBytes))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            if downloadingID == artifact.id {
                ProgressView(value: downloadProgress)
                    .frame(width: 100)
                    .progressViewStyle(.linear)
            } else if downloadedSet.contains(artifact.id) {
                HStack(spacing: 8) {
                    Label("Downloaded", systemImage: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                        .font(.callout)

                    Button(role: .destructive) {
                        deleteModel(artifact)
                    } label: {
                        Image(systemName: "trash")
                    }
                    .buttonStyle(.borderless)
                    .help("Delete model")
                }
            } else {
                Button("Download") {
                    Task { await downloadModel(artifact) }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
    }

    private func refreshStatus() async {
        var downloaded: Set<String> = []
        for artifact in artifacts {
            if await manager.isDownloaded(artifact) {
                downloaded.insert(artifact.id)
            }
        }
        downloadedSet = downloaded
    }

    private func downloadModel(_ artifact: ModelArtifact) async {
        errorMessage = nil
        downloadingID = artifact.id
        downloadProgress = 0
        do {
            _ = try await manager.download(artifact)
            downloadedSet.insert(artifact.id)
        } catch {
            errorMessage = "Download failed: \(error.localizedDescription)"
        }
        downloadingID = nil
    }

    private func deleteModel(_ artifact: ModelArtifact) {
        let path = FileUtilities.modelDirectory.appendingPathComponent(artifact.name)
        try? FileManager.default.removeItem(at: path)
        downloadedSet.remove(artifact.id)
    }

    static func formatBytes(_ bytes: Int64) -> String {
        let gb = Double(bytes) / 1_073_741_824
        if gb >= 1 { return String(format: "%.1f GB", gb) }
        let mb = Double(bytes) / 1_048_576
        return String(format: "%.0f MB", mb)
    }

    static let defaultArtifacts: [ModelArtifact] = [
        ModelArtifact(
            name: "ace-step-v1-3.5B-fp16.safetensors",
            downloadURL: URL(string: "https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B/resolve/main/model-fp16.safetensors")!,
            sha256: "",
            sizeBytes: 7_000_000_000
        ),
        ModelArtifact(
            name: "ace-step-v1-3.5B-int8.safetensors",
            downloadURL: URL(string: "https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B/resolve/main/model-int8.safetensors")!,
            sha256: "",
            sizeBytes: 3_500_000_000
        ),
    ]
}

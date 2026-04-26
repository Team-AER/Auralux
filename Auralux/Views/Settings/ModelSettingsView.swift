import SwiftUI

struct ModelSettingsView: View {
    @Environment(EngineService.self) private var engine
    @State private var modelStatus: ModelStatus = .unknown
    @State private var isDownloading = false
    @State private var errorMessage: String?
    @State private var serverRunning = false

    private let manager = ModelManagerService()

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Models")
                .font(.headline)

            engineStatusSection

            if let errorMessage {
                Label(errorMessage, systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.red)
                    .font(.callout)
            }

            modelListSection

            actionsSection
        }
        .task {
            await refreshStatus()
        }
    }

    // MARK: - Engine Status

    @ViewBuilder
    private var engineStatusSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Circle()
                    .fill(serverRunning ? (modelStatus.ditLoaded ? .green : .yellow) : .red)
                    .frame(width: 8, height: 8)

                if serverRunning {
                    if modelStatus.ditLoaded {
                        Text("Engine ready — \(modelStatus.engine)")
                            .font(.callout)
                            .foregroundStyle(.secondary)
                    } else if isDownloading {
                        downloadProgressLabel
                    } else {
                        Text("Server running, models not loaded")
                            .font(.callout)
                            .foregroundStyle(.secondary)
                    }
                } else {
                    Text("Server not running")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
            }

            if serverRunning && modelStatus.ditLoaded {
                HStack(spacing: 16) {
                    Label(modelStatus.device, systemImage: "cpu")
                    if !modelStatus.ditModel.isEmpty {
                        Label(modelStatus.ditModel, systemImage: "waveform")
                    }
                    if modelStatus.llmLoaded, !modelStatus.llmModel.isEmpty {
                        Label(modelStatus.llmModel, systemImage: "brain")
                    }
                }
                .font(.caption)
                .foregroundStyle(.tertiary)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 8))
    }

    // MARK: - Download Progress

    @ViewBuilder
    private var downloadProgressLabel: some View {
        if case .settingUp(let progress) = engine.state {
            VStack(alignment: .leading, spacing: 4) {
                Text(progress)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                if let pct = parsePercent(from: progress) {
                    ProgressView(value: Double(pct), total: 100)
                        .frame(width: 160)
                        .controlSize(.small)
                }
            }
        } else {
            HStack(spacing: 6) {
                ProgressView().controlSize(.small)
                Text("Downloading models …")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func parsePercent(from string: String) -> Int? {
        guard let range = string.range(of: #"\d+"#, options: .regularExpression),
              let value = Int(string[range]) else { return nil }
        return value
    }

    // MARK: - Model List

    @ViewBuilder
    private var modelListSection: some View {
        VStack(spacing: 0) {
            ForEach(ModelManagerService.knownArtifacts) { artifact in
                modelRow(artifact)
                if artifact.id != ModelManagerService.knownArtifacts.last?.id {
                    Divider()
                }
            }
        }
        .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 8))
    }

    @ViewBuilder
    private func modelRow(_ artifact: ModelArtifact) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(artifact.name)
                    .font(.body.weight(.medium))
                Text(artifact.description)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }

            Spacer()

            VStack(alignment: .trailing, spacing: 4) {
                Text(String(format: "~%.1f GB", artifact.estimatedSizeGB))
                    .font(.caption)
                    .foregroundStyle(.tertiary)

                statusBadge(for: artifact)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
    }

    @ViewBuilder
    private func statusBadge(for artifact: ModelArtifact) -> some View {
        if !serverRunning {
            Text("Offline")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(.quaternary, in: Capsule())
        } else if isModelLoaded(artifact) {
            Label("Loaded", systemImage: "checkmark.circle.fill")
                .font(.caption)
                .foregroundStyle(.green)
        } else if isDownloading {
            ProgressView()
                .controlSize(.small)
        } else {
            Text("Not loaded")
                .font(.caption2)
                .foregroundStyle(.orange)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(.orange.opacity(0.1), in: Capsule())
        }
    }

    private func isModelLoaded(_ artifact: ModelArtifact) -> Bool {
        switch artifact.name {
        case _ where artifact.name.contains("turbo") || artifact.name.contains("base") || artifact.name.contains("sft"):
            return modelStatus.ditLoaded
        case _ where artifact.name.contains("lm"):
            return modelStatus.llmLoaded
        case "vae":
            return modelStatus.ditLoaded
        default:
            return false
        }
    }

    // MARK: - Actions

    @ViewBuilder
    private var actionsSection: some View {
        HStack {
            if serverRunning && !modelStatus.ditLoaded && !isDownloading {
                Button("Download Models") {
                    Task { await downloadModels() }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
            }

            Spacer()

            Button("Refresh") {
                Task { await refreshStatus() }
            }
            .controlSize(.small)
        }
        .font(.callout)

        Text("Models are auto-downloaded from HuggingFace on first generation. Stored in the ACE-Step checkpoints directory.")
            .font(.caption2)
            .foregroundStyle(.tertiary)
    }

    // MARK: - Networking

    private func refreshStatus() async {
        serverRunning = await manager.isServerHealthy()
        let status = await manager.fetchModelStatus()
        modelStatus = status
        errorMessage = status.error
    }

    private func downloadModels() async {
        isDownloading = true
        errorMessage = nil
        do {
            try await manager.triggerModelDownload()
            try? await Task.sleep(for: .seconds(2))
            await refreshStatus()
        } catch {
            errorMessage = "Download failed: \(error.localizedDescription)"
        }
        isDownloading = false
    }
}

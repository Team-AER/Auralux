import SwiftUI

struct ModelSettingsView: View {
    @Environment(NativeInferenceEngine.self) private var engine

    private var isLowMemoryMac: Bool {
        ProcessInfo.processInfo.physicalMemory <= 17_179_869_184
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Models")
                .font(.headline)

            modelStateSection
            modelListSection
            footerSection
        }
    }

    // MARK: - Model State Section

    @ViewBuilder
    private var modelStateSection: some View {
        HStack(alignment: .top, spacing: 8) {
            Circle()
                .fill(statusColor)
                .frame(width: 8, height: 8)
                .padding(.top, 3)

            VStack(alignment: .leading, spacing: 4) {
                Text(statusTitle)
                    .font(.callout)

                switch engine.modelState {
                case .notDownloaded:
                    Text("Open setup to download model weights automatically (~5.4 GB).")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                case .downloading(let progress):
                    VStack(alignment: .leading, spacing: 4) {
                        ProgressView(value: progress)
                            .progressViewStyle(.linear)
                        Text("Downloading \(Int(progress * 100))% — ~5.4 GB total")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                case .error(let message):
                    Text(message)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .lineLimit(3)
                case .loading:
                    HStack(spacing: 6) {
                        ProgressView().controlSize(.small)
                        Text("Loading...")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                case .ready:
                    if isLowMemoryMac {
                        Label("16 GB Mac — running in memory-efficient mode", systemImage: "memorychip")
                            .font(.caption)
                            .foregroundStyle(.orange.opacity(0.9))
                    }
                }
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 8))
    }

    private var statusColor: Color {
        switch engine.modelState {
        case .ready:        return .green
        case .loading:      return .orange
        case .downloading:  return .blue
        case .notDownloaded: return .gray
        case .error:        return .red
        }
    }

    private var statusTitle: String {
        switch engine.modelState {
        case .ready:                    return "Models loaded — ready to generate"
        case .loading:                  return "Loading model weights…"
        case .downloading(let p):       return "Downloading model weights (\(Int(p * 100))%)"
        case .notDownloaded:            return "Weights not found"
        case .error:                    return "Failed to load models"
        }
    }

    // MARK: - Model List

    @ViewBuilder
    private var modelListSection: some View {
        VStack(spacing: 0) {
            let artifacts = ModelManagerService.mlxArtifacts
            ForEach(artifacts) { artifact in
                modelRow(artifact)
                if artifact.id != artifacts.last?.id {
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

            Text(String(format: "~%.1f GB", artifact.estimatedSizeGB))
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
    }

    // MARK: - Actions

    @ViewBuilder
    private var footerSection: some View {
        HStack {
            switch engine.modelState {
            case .error:
                Button("Retry Load") {
                    Task { await engine.loadModels() }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
            case .notDownloaded:
                Text("python tools/convert_weights.py")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 4))
            default:
                EmptyView()
            }
            Spacer()
        }

        Text("Weights stored in ~/Library/Application Support/Auralux/Models/ace-step-v1.5-mlx/")
            .font(.caption2)
            .foregroundStyle(.tertiary)
            .textSelection(.enabled)
    }
}

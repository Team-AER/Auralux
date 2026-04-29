import SwiftUI

struct ModelSettingsView: View {
    @Environment(NativeInferenceEngine.self) private var engine
    @Environment(SettingsViewModel.self) private var settings

    @State private var downloadSheetVariant: DiTVariant? = nil

    private var isLowMemoryMac: Bool {
        ProcessInfo.processInfo.physicalMemory <= 17_179_869_184
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Models")
                .font(.headline)

            loadedModelStatus
            variantList
            pathFooter
        }
        .sheet(item: $downloadSheetVariant) { variant in
            ModelDownloadSheet(variant: variant)
                .environment(engine)
        }
    }

    // MARK: - Loaded model status

    @ViewBuilder
    private var loadedModelStatus: some View {
        HStack(alignment: .top, spacing: 8) {
            Circle()
                .fill(statusColor)
                .frame(width: 8, height: 8)
                .padding(.top, 4)

            VStack(alignment: .leading, spacing: 4) {
                Text(statusTitle)
                    .font(.callout)

                switch engine.modelState {
                case .notDownloaded:
                    Text("Select a model below and click Download.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                case .downloaded:
                    Text("Models are unloaded. They will load automatically when you generate.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                case .downloading(let progress):
                    VStack(alignment: .leading, spacing: 4) {
                        ProgressView(value: progress)
                            .progressViewStyle(.linear)
                        Text("Downloading \(Int(progress * 100))%")
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
                        Text("Loading weights into memory…")
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
        case .ready:         return .green
        case .downloaded:    return .green.opacity(0.6)
        case .loading:       return .orange
        case .downloading:   return .blue
        case .notDownloaded: return .gray
        case .error:         return .red
        }
    }

    private var statusTitle: String {
        switch engine.modelState {
        case .ready:                    return "Models loaded — ready to generate"
        case .downloaded:               return "Ready — models load on first generate"
        case .loading:                  return "Loading model weights…"
        case .downloading(let p):       return "Downloading (\(Int(p * 100))%)"
        case .notDownloaded:            return "No model loaded"
        case .error:                    return "Failed to load models"
        }
    }

    // MARK: - Variant list

    private var variantList: some View {
        VStack(spacing: 0) {
            let variants = DiTVariant.allCases.filter(\.isAvailable)
            ForEach(variants) { variant in
                variantRow(variant)
                if variant.id != variants.last?.id {
                    Divider()
                }
            }
        }
        .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 8))
    }

    @ViewBuilder
    private func variantRow(_ variant: DiTVariant) -> some View {
        let downloaded = engine.isDownloaded(variant)
        let isDownloading = engine.activeDownloadVariant == variant
        let isActive = settings.ditVariant == variant && engine.modelState.isReady

        HStack(spacing: 12) {
            // Status dot
            Circle()
                .fill(rowDotColor(variant: variant, downloaded: downloaded,
                                  downloading: isDownloading, active: isActive))
                .frame(width: 7, height: 7)

            // Name + description
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(variant.displayName)
                        .font(.body.weight(.medium))
                    if isActive {
                        Text("In Use")
                            .font(.caption2.weight(.semibold))
                            .padding(.horizontal, 5)
                            .padding(.vertical, 2)
                            .background(.green.opacity(0.15), in: Capsule())
                            .foregroundStyle(.green)
                    }
                }
                Text(rowSubtitle(variant: variant, downloaded: downloaded, downloading: isDownloading))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            Spacer()

            // Size + action button
            if isDownloading {
                downloadingIndicator(variant: variant)
            } else if downloaded {
                Text(formattedSize(bytes: ModelDownloader.estimatedBytes(for: variant)))
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            } else {
                downloadButton(variant: variant)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .contentShape(Rectangle())
    }

    private func rowDotColor(variant: DiTVariant, downloaded: Bool, downloading: Bool, active: Bool) -> Color {
        if active       { return .green }
        if downloading  { return .blue }
        if downloaded   { return .green.opacity(0.5) }
        return .gray.opacity(0.4)
    }

    private func rowSubtitle(variant: DiTVariant, downloaded: Bool, downloading: Bool) -> String {
        if downloading {
            return "Downloading \(Int(engine.downloadProgress * 100))%…"
        }
        if downloaded {
            return "Downloaded — \(formattedSize(bytes: ModelDownloader.estimatedBytes(for: variant)))"
        }
        if !variant.canDownloadInApp {
            return "Requires conversion script"
        }
        if variant.requiresTurboBase && !engine.isDownloaded(.turbo) {
            return "Requires Turbo base first — \(formattedSize(bytes: ModelDownloader.estimatedBytes(for: variant)))"
        }
        return "Not downloaded — \(formattedSize(bytes: ModelDownloader.estimatedBytes(for: variant)))"
    }

    @ViewBuilder
    private func downloadingIndicator(variant: DiTVariant) -> some View {
        HStack(spacing: 6) {
            ProgressView().controlSize(.mini)
            Text("\(Int(engine.downloadProgress * 100))%")
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private func downloadButton(variant: DiTVariant) -> some View {
        Button {
            downloadSheetVariant = variant
        } label: {
            Image(systemName: "arrow.down.circle")
                .font(.body)
        }
        .buttonStyle(.plain)
        .foregroundStyle(.tint)
        .help(variant.canDownloadInApp ? "Download \(variant.displayName)" : "Requires conversion script")
    }

    // MARK: - Footer

    @ViewBuilder
    private var pathFooter: some View {
        if case .error = engine.modelState {
            HStack {
                Button("Retry Load") {
                    Task { await engine.loadModels() }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                Spacer()
            }
        }

        Text("~/Library/Application Support/Auralux/Models/")
            .font(.caption2)
            .foregroundStyle(.tertiary)
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Helpers

    private func formattedSize(bytes: Int64) -> String {
        guard bytes > 0 else { return "—" }
        let gb = Double(bytes) / 1_000_000_000
        return String(format: "~%.1f GB", gb)
    }
}

import SwiftUI

/// Per-variant download sheet — presented from ModelSettingsView or the
/// model-not-ready banner in GenerationView.
struct ModelDownloadSheet: View {
    let variant: DiTVariant

    @Environment(NativeInferenceEngine.self) private var engine
    @Environment(\.dismiss) private var dismiss

    private var isThisVariantDownloading: Bool {
        engine.activeDownloadVariant == variant
    }
    private var isAlreadyDownloaded: Bool {
        engine.isDownloaded(variant)
    }
    private var turboMissing: Bool {
        variant.requiresTurboBase && !engine.isDownloaded(.turbo)
    }

    // MARK: - Body

    var body: some View {
        VStack(spacing: 0) {
            headerSection
            Divider().opacity(0.4)
            contentSection
            Divider().opacity(0.4)
            footerSection
        }
        .frame(width: 420)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
        .shadow(color: .black.opacity(0.18), radius: 28, y: 10)
    }

    // MARK: - Header

    private var headerSection: some View {
        VStack(spacing: 8) {
            Image(systemName: headerIcon)
                .font(.system(size: 34, weight: .medium))
                .foregroundStyle(headerIconColor)
                .padding(.top, 24)

            Text(headerTitle)
                .font(.headline)

            Text(variant.description(showSteps: true))
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 28)
        }
        .padding(.bottom, 20)
    }

    private var headerIcon: String {
        if isAlreadyDownloaded        { return "checkmark.circle.fill" }
        if !variant.canDownloadInApp  { return "terminal" }
        if turboMissing               { return "exclamationmark.triangle" }
        if isThisVariantDownloading   { return "arrow.down.circle" }
        return "arrow.down.circle"
    }

    private var headerIconColor: Color {
        if isAlreadyDownloaded        { return .green }
        if turboMissing               { return .orange }
        if !variant.canDownloadInApp  { return .secondary }
        return .accentColor
    }

    private var headerTitle: String {
        if isAlreadyDownloaded        { return "\(variant.displayName) — Ready" }
        if !variant.canDownloadInApp  { return "Script Conversion Required" }
        if turboMissing               { return "Turbo Base Required" }
        if isThisVariantDownloading   { return "Downloading…" }
        return "Download \(variant.displayName)"
    }

    // MARK: - Content

    @ViewBuilder
    private var contentSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            if !variant.canDownloadInApp {
                scriptRequiredContent
            } else if turboMissing {
                turboRequiredContent
            } else if isAlreadyDownloaded {
                alreadyDownloadedContent
            } else {
                downloadInfoContent
            }
        }
        .padding(24)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // Script-only variants (XL)
    private var scriptRequiredContent: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(
                "XL variants require converting upstream PyTorch checkpoints. Run the conversion script after downloading the Turbo base.",
                systemImage: "info.circle"
            )
            .font(.callout)
            .foregroundStyle(.secondary)

            VStack(alignment: .leading, spacing: 6) {
                Text("Conversion command")
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)
                Text("python tools/convert_weights.py --variant \(variant.rawValue)")
                    .font(.caption.monospaced())
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 6))
                    .textSelection(.enabled)
            }
        }
    }

    // Turbo not downloaded yet
    private var turboRequiredContent: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(
                "SFT and Base variants share the language model, VAE decoder, and text encoder from the Turbo package. Turbo must be downloaded first.",
                systemImage: "link"
            )
            .font(.callout)
            .foregroundStyle(.secondary)

            HStack(spacing: 10) {
                ForEach(["LM (1.3 GB)", "VAE (169 MB)", "Text Encoder (~1.2 GB)"], id: \.self) { label in
                    Text(label)
                        .font(.caption)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(.quaternary.opacity(0.5), in: Capsule())
                }
            }
        }
    }

    // Already downloaded
    private var alreadyDownloadedContent: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("All weight files are present on disk.", systemImage: "checkmark.circle")
                .foregroundStyle(.green)
                .font(.callout)
            Text("Weights stored in ~/Library/Application Support/Auralux/Models/\(variant.mlxDirectoryName)/")
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .textSelection(.enabled)
        }
    }

    // Normal download info + optional progress
    private var downloadInfoContent: some View {
        VStack(alignment: .leading, spacing: 14) {
            componentList

            if isThisVariantDownloading {
                downloadProgressSection
            } else {
                sizeFooter
            }
        }
    }

    private var componentList: some View {
        VStack(spacing: 0) {
            if variant == .turbo {
                componentRow(icon: "cpu", label: "DiT Weights",    detail: "~3.9 GB")
                componentRow(icon: "waveform",  label: "VAE Decoder",    detail: "~169 MB", divider: true)
                componentRow(icon: "brain",     label: "Text Encoder",   detail: "~1.2 GB", divider: true)
                componentRow(icon: "music.note",label: "Language Model", detail: "~1.3 GB", divider: true)
            } else {
                componentRow(icon: "cpu",   label: "DiT Weights",       detail: "~4.8 GB")
                componentRow(icon: "link",  label: "Shared Components", detail: "via Turbo", divider: true)
            }
        }
        .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 8))
    }

    private func componentRow(icon: String, label: String, detail: String, divider: Bool = false) -> some View {
        VStack(spacing: 0) {
            if divider { Divider() }
            HStack {
                Image(systemName: icon)
                    .frame(width: 18)
                    .foregroundStyle(.secondary)
                Text(label)
                    .font(.callout)
                Spacer()
                Text(detail)
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 9)
        }
    }

    private var downloadProgressSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            ProgressView(value: engine.downloadProgress)
                .progressViewStyle(.linear)
                .accentColor(.accentColor)

            HStack {
                Text("\(Int(engine.downloadProgress * 100))% downloaded")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Spacer()
                Text(formattedSize(bytes: ModelDownloader.estimatedBytes(for: variant)))
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
    }

    private var sizeFooter: some View {
        HStack {
            Image(systemName: "internaldrive")
                .foregroundStyle(.secondary)
                .font(.caption)
            Text("Total download: \(formattedSize(bytes: ModelDownloader.estimatedBytes(for: variant)))")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - Footer

    private var footerSection: some View {
        HStack {
            Button("Close") { dismiss() }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)

            Spacer()

            if !variant.canDownloadInApp {
                EmptyView()
            } else if turboMissing {
                Button("Download Turbo First") {
                    dismiss()
                    Task { await engine.download(.turbo) }
                }
                .buttonStyle(.borderedProminent)
            } else if isAlreadyDownloaded {
                EmptyView()
            } else if isThisVariantDownloading {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text("Downloading…")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
            } else {
                Button("Download") {
                    Task { await engine.download(variant) }
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 14)
        .animation(.default, value: isThisVariantDownloading)
    }

    // MARK: - Helpers

    private func formattedSize(bytes: Int64) -> String {
        let gb = Double(bytes) / 1_000_000_000
        return String(format: "~%.1f GB", gb)
    }
}

// MARK: - DiTVariant display helper (local)

private extension DiTVariant {
    func description(showSteps: Bool) -> String {
        switch self {
        case .turbo:   return "ACE-Step v1.5 — 8-step CFG-distilled, MLX-native (DiT 2B + LM 0.6B)"
        case .sft:     return "ACE-Step v1.5 — 60-step flow-matching, MLX-native (DiT 2B, shares base)"
        case .base:    return "ACE-Step v1.5 — 60-step flow-matching, MLX-native (DiT 2B, shares base)"
        case .xlTurbo: return "ACE-Step v1.5 — 8-step CFG-distilled, MLX-native (DiT 5B + LM 0.6B)"
        case .xlSft:   return "ACE-Step v1.5 — 60-step flow-matching, MLX-native (DiT 5B, shares base)"
        case .xlBase:  return "ACE-Step v1.5 — 60-step flow-matching, MLX-native (DiT 5B, shares base)"
        }
    }
}

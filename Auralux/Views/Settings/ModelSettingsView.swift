import SwiftUI

struct ModelSettingsView: View {
    @Environment(NativeInferenceEngine.self) private var engine
    @Environment(SettingsViewModel.self) private var settings

    @State private var downloadSheetVariant: DiTVariant? = nil
    @State private var pendingDelete: DiTVariant? = nil
    @State private var pendingRedownload: DiTVariant? = nil
    @State private var showAddCustomSheet = false
    @State private var pendingCustomDelete: CustomModel? = nil
    @State private var pendingCustomRedownload: CustomModel? = nil

    private var isLowMemoryMac: Bool { AppConstants.isLowMemoryMachine }

    var body: some View {
        contentWithSheets
            .modifier(CustomModelDialogs(
                pendingDelete: $pendingCustomDelete,
                pendingRedownload: $pendingCustomRedownload,
                engine: engine,
                deleteMessage: customDeleteMessage(for:)
            ))
            .modifier(VariantModelDialogs(
                pendingDelete: $pendingDelete,
                pendingRedownload: $pendingRedownload,
                engine: engine,
                deleteMessage: deleteMessage(for:),
                redownloadMessage: redownloadMessage(for:)
            ))
    }

    private var contentWithSheets: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Models")
                    .font(.headline)
                Spacer()
                Button {
                    showAddCustomSheet = true
                } label: {
                    Label("Add Model", systemImage: "plus")
                        .font(.callout)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }

            loadedModelStatus
            variantList
            if !engine.customModels.models.isEmpty {
                customList
            }
            pathFooter
        }
        .sheet(isPresented: $showAddCustomSheet) {
            AddCustomModelSheet()
                .environment(engine)
                .environment(settings)
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

    private var customList: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Custom Models")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.secondary)
            VStack(spacing: 0) {
                let models = engine.customModels.models
                ForEach(models) { model in
                    customRow(model)
                    if model.id != models.last?.id {
                        Divider()
                    }
                }
            }
            .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 8))
        }
    }

    @ViewBuilder
    private func variantRow(_ variant: DiTVariant) -> some View {
        let downloaded = engine.isDownloaded(variant)
        let isDownloading = engine.activeDownloadVariant == variant
        let isActive = settings.activeCustomModelID == nil
            && settings.ditVariant == variant
            && engine.modelState.isReady

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
                downloadedActions(variant: variant)
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
    private func downloadedActions(variant: DiTVariant) -> some View {
        HStack(spacing: 8) {
            Text(formattedSize(bytes: ModelDownloader.estimatedBytes(for: variant)))
                .font(.caption)
                .foregroundStyle(.tertiary)

            Menu {
                Button {
                    pendingRedownload = variant
                } label: {
                    Label("Redownload", systemImage: "arrow.clockwise")
                }
                Button(role: .destructive) {
                    pendingDelete = variant
                } label: {
                    Label("Delete", systemImage: "trash")
                }
            } label: {
                Image(systemName: "ellipsis.circle")
                    .font(.body)
                    .foregroundStyle(.secondary)
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .fixedSize()
            .help("Manage \(variant.displayName)")
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

    // MARK: - Custom row

    @ViewBuilder
    private func customRow(_ model: CustomModel) -> some View {
        let downloaded = engine.isDownloaded(model)
        let isDownloading = engine.activeDownloadCustomID == model.id
        let isActive = settings.activeCustomModelID == model.id && engine.modelState.isReady

        HStack(spacing: 12) {
            Circle()
                .fill(customRowDotColor(downloaded: downloaded, downloading: isDownloading, active: isActive))
                .frame(width: 7, height: 7)

            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(model.displayName)
                        .font(.body.weight(.medium))
                    if isActive {
                        Text("In Use")
                            .font(.caption2.weight(.semibold))
                            .padding(.horizontal, 5)
                            .padding(.vertical, 2)
                            .background(.green.opacity(0.15), in: Capsule())
                            .foregroundStyle(.green)
                    }
                    Text("base: \(model.baseVariant.rawValue)")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 2)
                        .background(.quaternary.opacity(0.4), in: Capsule())
                }
                Text(customRowSubtitle(model: model, downloaded: downloaded, downloading: isDownloading))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }

            Spacer()

            if isDownloading {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.mini)
                    Text("\(Int(engine.downloadProgress * 100))%")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            } else if downloaded {
                customDownloadedActions(model)
            } else {
                Button {
                    Task { await engine.downloadCustom(model) }
                } label: {
                    Image(systemName: "arrow.down.circle")
                        .font(.body)
                }
                .buttonStyle(.plain)
                .foregroundStyle(.tint)
                .help("Download \(model.displayName)")
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .contentShape(Rectangle())
        .onTapGesture {
            // Click anywhere on the row to make this model active.
            settings.ditVariant = model.baseVariant
            settings.activeCustomModelID = model.id
            engine.unloadModels()
        }
    }

    private func customRowDotColor(downloaded: Bool, downloading: Bool, active: Bool) -> Color {
        if active      { return .green }
        if downloading { return .blue }
        if downloaded  { return .green.opacity(0.5) }
        return .gray.opacity(0.4)
    }

    private func customRowSubtitle(model: CustomModel, downloaded: Bool, downloading: Bool) -> String {
        if downloading {
            return "Downloading \(Int(engine.downloadProgress * 100))%…"
        }
        if downloaded {
            return "Ready — \(model.sourceDescription)"
        }
        if case .localFolder = model.source {
            return "Folder missing required files — \(model.sourceDescription)"
        }
        return "Not downloaded — \(model.sourceDescription)"
    }

    @ViewBuilder
    private func customDownloadedActions(_ model: CustomModel) -> some View {
        Menu {
            Button {
                settings.ditVariant = model.baseVariant
                settings.activeCustomModelID = model.id
                engine.unloadModels()
            } label: {
                Label("Use This Model", systemImage: "checkmark.circle")
            }
            if case .huggingFace = model.source {
                Button {
                    pendingCustomRedownload = model
                } label: {
                    Label("Redownload", systemImage: "arrow.clockwise")
                }
            }
            Button(role: .destructive) {
                pendingCustomDelete = model
            } label: {
                Label(model.isHFManaged ? "Delete" : "Remove", systemImage: "trash")
            }
        } label: {
            Image(systemName: "ellipsis.circle")
                .font(.body)
                .foregroundStyle(.secondary)
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("Manage \(model.displayName)")
    }

    private func customDeleteMessage(for model: CustomModel) -> String {
        switch model.source {
        case .huggingFace:
            return "Removes \(model.displayName) weights from disk and unregisters it."
        case .localFolder:
            return "Unregisters \(model.displayName). The original folder is left untouched."
        }
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

    private func deleteMessage(for variant: DiTVariant) -> String {
        let size = formattedSize(bytes: ModelDownloader.estimatedBytes(for: variant))
        if variant == .turbo {
            // SFT/Base symlink lm/, vae/, text/ from Turbo — they break if Turbo is removed.
            let dependents = DiTVariant.allCases.filter { $0.requiresTurboBase && engine.isDownloaded($0) }
            if !dependents.isEmpty {
                let names = dependents.map(\.displayName).joined(separator: ", ")
                return "Removes Turbo (\(size)) and the variants that share its components: \(names). You'll need to redownload them separately."
            }
        }
        return "Removes \(variant.displayName) weights from disk (\(size) freed)."
    }

    private func redownloadMessage(for variant: DiTVariant) -> String {
        let size = formattedSize(bytes: ModelDownloader.estimatedBytes(for: variant))
        if variant == .turbo {
            let dependents = DiTVariant.allCases.filter { $0.requiresTurboBase && engine.isDownloaded($0) }
            if !dependents.isEmpty {
                let names = dependents.map(\.displayName).joined(separator: ", ")
                return "Deletes Turbo and its dependent variants (\(names)), then redownloads Turbo (\(size)). Other variants must be redownloaded manually."
            }
        }
        return "Deletes and redownloads \(variant.displayName) (\(size))."
    }
}

private struct CustomModelDialogs: ViewModifier {
    @Binding var pendingDelete: CustomModel?
    @Binding var pendingRedownload: CustomModel?
    let engine: NativeInferenceEngine
    let deleteMessage: (CustomModel) -> String

    func body(content: Content) -> some View {
        content
            .confirmationDialog(
                pendingDelete.map { "Delete \($0.displayName)?" } ?? "",
                isPresented: Binding(
                    get: { pendingDelete != nil },
                    set: { if !$0 { pendingDelete = nil } }
                ),
                presenting: pendingDelete
            ) { model in
                Button("Delete", role: .destructive) {
                    Task { await engine.deleteCustom(model) }
                }
                Button("Cancel", role: .cancel) { }
            } message: { model in
                Text(deleteMessage(model))
            }
            .confirmationDialog(
                pendingRedownload.map { "Redownload \($0.displayName)?" } ?? "",
                isPresented: Binding(
                    get: { pendingRedownload != nil },
                    set: { if !$0 { pendingRedownload = nil } }
                ),
                presenting: pendingRedownload
            ) { model in
                Button("Redownload") {
                    Task { await engine.redownloadCustom(model) }
                }
                Button("Cancel", role: .cancel) { }
            } message: { model in
                Text("Deletes and redownloads \(model.displayName).")
            }
    }
}

private struct VariantModelDialogs: ViewModifier {
    @Binding var pendingDelete: DiTVariant?
    @Binding var pendingRedownload: DiTVariant?
    let engine: NativeInferenceEngine
    let deleteMessage: (DiTVariant) -> String
    let redownloadMessage: (DiTVariant) -> String

    func body(content: Content) -> some View {
        content
            .confirmationDialog(
                pendingDelete.map { "Delete \($0.displayName)?" } ?? "",
                isPresented: Binding(
                    get: { pendingDelete != nil },
                    set: { if !$0 { pendingDelete = nil } }
                ),
                presenting: pendingDelete
            ) { variant in
                Button("Delete", role: .destructive) {
                    Task { await engine.delete(variant) }
                }
                Button("Cancel", role: .cancel) { }
            } message: { variant in
                Text(deleteMessage(variant))
            }
            .confirmationDialog(
                pendingRedownload.map { "Redownload \($0.displayName)?" } ?? "",
                isPresented: Binding(
                    get: { pendingRedownload != nil },
                    set: { if !$0 { pendingRedownload = nil } }
                ),
                presenting: pendingRedownload
            ) { variant in
                Button("Redownload") {
                    Task { await engine.redownload(variant) }
                }
                Button("Cancel", role: .cancel) { }
            } message: { variant in
                Text(redownloadMessage(variant))
            }
    }
}

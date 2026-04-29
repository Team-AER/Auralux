import SwiftUI
import UniformTypeIdentifiers
import AppKit

/// Sheet for adding a custom ACE-Step-compatible model — either by HuggingFace
/// repo URL (downloaded into the app's Models directory) or by browsing to a
/// local folder (registered in place, no copy).
struct AddCustomModelSheet: View {
    @Environment(NativeInferenceEngine.self) private var engine
    @Environment(SettingsViewModel.self) private var settings
    @Environment(\.dismiss) private var dismiss

    enum Source: String, CaseIterable, Identifiable {
        case huggingFace = "Hugging Face"
        case localFolder = "Local Folder"
        var id: String { rawValue }
    }

    @State private var source: Source = .huggingFace
    @State private var hfURL: String = ""
    @State private var localFolder: URL? = nil
    @State private var displayName: String = ""
    @State private var baseVariant: DiTVariant = .turbo
    @State private var errorMessage: String? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            header
            Divider().opacity(0.4)
            content
            if let errorMessage {
                Label(errorMessage, systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.red)
                    .font(.caption)
                    .padding(.horizontal, 24)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Divider().opacity(0.4)
            footer
        }
        .frame(width: 460)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
        .shadow(color: .black.opacity(0.18), radius: 28, y: 10)
    }

    private var header: some View {
        VStack(spacing: 8) {
            Image(systemName: "plus.app")
                .font(.system(size: 30, weight: .medium))
                .foregroundStyle(.tint)
                .padding(.top, 22)
            Text("Add Custom Model")
                .font(.headline)
            Text("ACE-Step-compatible weights, MLX format. Inherits step count and CFG behavior from a base variant.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 28)
        }
    }

    @ViewBuilder
    private var content: some View {
        VStack(alignment: .leading, spacing: 14) {
            Picker("Source", selection: $source) {
                ForEach(Source.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()

            switch source {
            case .huggingFace: hfFields
            case .localFolder: localFields
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Display Name")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                TextField("e.g. ACE-Step Lo-Fi finetune", text: $displayName)
                    .textFieldStyle(.roundedBorder)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Base Variant")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Picker("", selection: $baseVariant) {
                    ForEach(DiTVariant.allCases.filter { $0.canDownloadInApp }) { v in
                        Text(v.displayName).tag(v)
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
                Text(baseVariantHint)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(.horizontal, 24)
    }

    @ViewBuilder
    private var hfFields: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Hugging Face Repo URL")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            TextField("https://huggingface.co/owner/repo", text: $hfURL)
                .textFieldStyle(.roundedBorder)
            Text("The repo must follow the ACE-Step MLX layout (dit/, lm/, vae/, text/, silence_latent.safetensors).")
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder
    private var localFields: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Model Folder")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            HStack {
                Text(localFolder?.path ?? "No folder selected")
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .frame(maxWidth: .infinity, alignment: .leading)
                Button("Browse…") { browseForFolder() }
            }
            .padding(8)
            .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 6))
        }
    }

    private var baseVariantHint: String {
        switch baseVariant {
        case .turbo: return "8-step CFG-distilled. Custom weights need lm/, vae/, text/ included."
        case .sft, .base: return "60-step. Custom weights only need dit/. lm/, vae/, text/ are linked from Turbo (which must already be downloaded)."
        default: return ""
        }
    }

    private var footer: some View {
        HStack {
            Button("Cancel") { dismiss() }
                .keyboardShortcut(.cancelAction)
            Spacer()
            Button(primaryButtonTitle) { confirm() }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(!isReady)
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 14)
    }

    private var primaryButtonTitle: String {
        source == .huggingFace ? "Add & Download" : "Import"
    }

    private var isReady: Bool {
        switch source {
        case .huggingFace:
            return HuggingFaceURL.parseRepoID(hfURL) != nil
        case .localFolder:
            return localFolder != nil
        }
    }

    private func browseForFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.title = "Select model folder"
        if panel.runModal() == .OK, let url = panel.url {
            localFolder = url
            if displayName.isEmpty { displayName = url.lastPathComponent }
        }
    }

    private func confirm() {
        errorMessage = nil
        switch source {
        case .huggingFace:
            guard let repoID = HuggingFaceURL.parseRepoID(hfURL) else {
                errorMessage = "Couldn't parse a Hugging Face repo from that URL."
                return
            }
            let model = CustomModel(
                id: UUID().uuidString,
                displayName: displayName.isEmpty ? repoID : displayName,
                source: .huggingFace(repoID: repoID),
                baseVariant: baseVariant,
                addedAt: Date()
            )
            engine.customModels.add(model)
            // Activate immediately so the download progress surfaces in the main status row.
            settings.ditVariant = baseVariant
            settings.activeCustomModelID = model.id
            engine.unloadModels()
            Task { await engine.downloadCustom(model) }
            dismiss()
        case .localFolder:
            guard let folder = localFolder else { return }
            do {
                let model = try engine.importLocalCustomModel(
                    displayName: displayName,
                    folder: folder,
                    baseVariant: baseVariant
                )
                settings.ditVariant = baseVariant
                settings.activeCustomModelID = model.id
                engine.unloadModels()
                dismiss()
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }
}

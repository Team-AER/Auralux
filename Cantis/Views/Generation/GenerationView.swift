import SwiftUI

struct GenerationView: View {
    @Environment(GenerationViewModel.self) private var viewModel
    @Environment(SettingsViewModel.self) private var settings
    @Environment(NativeInferenceEngine.self) private var engine
    @State private var tagText = ""
    @State private var didSeedDefaults = false
    @State private var downloadSheetVariant: DiTVariant? = nil

    private var engineReady: Bool {
        engine.modelState.isReady
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if !engineReady && !engine.isOnboarding && !engine.isGenerating {
                    modelBanner
                }

                GroupBox("Prompt") {
                    TextEditor(text: Bindable(viewModel).prompt)
                        .font(.body)
                        .frame(minHeight: 100)
                        .accessibilityLabel("Music prompt")
                        .accessibilityIdentifier("prompt-editor")
                }

                TagEditorView(tags: viewModel.tags, draftTag: $tagText) { tag in
                    viewModel.addTag(tag)
                } onRemoveTag: { tag in
                    viewModel.removeTag(tag)
                }

                LyricEditorView(lyrics: Bindable(viewModel).lyrics)
                ParameterControlsView()
            }
            .padding(20)
        }
        .onAppear {
            // Seed mode/numSteps/shift/cfg from saved settings the first time
            // the panel appears in this session. Per-request edits stay local
            // to the GenerationViewModel and don't write back to settings.
            guard !didSeedDefaults else { return }
            viewModel.applyDefaults(from: settings)
            didSeedDefaults = true
        }
        .sheet(item: $downloadSheetVariant) { variant in
            ModelDownloadSheet(variant: variant)
                .environment(engine)
        }
    }

    // MARK: - Model status banner

    @ViewBuilder
    private var modelBanner: some View {
        switch engine.modelState {
        case .notDownloaded:
            notDownloadedBanner

        case .downloaded:
            EmptyView()

        case .downloading(let progress):
            HStack(spacing: 12) {
                ProgressView().controlSize(.small)
                VStack(alignment: .leading, spacing: 3) {
                    Text("Downloading model weights…")
                        .font(.callout.bold())
                    ProgressView(value: progress)
                        .progressViewStyle(.linear)
                        .frame(maxWidth: 240)
                }
                Spacer()
                Text("\(Int(progress * 100))%")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            .padding(14)
            .background(.blue.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))

        case .loading:
            HStack(spacing: 12) {
                ProgressView().controlSize(.small)
                Text("Loading model into memory…")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                Spacer()
            }
            .padding(14)
            .background(.orange.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))

        case .error(let message):
            HStack(spacing: 12) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Model failed to load")
                        .font(.callout.bold())
                    Text(message)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
                Spacer()
                Button("Retry") {
                    Task { await engine.loadModels() }
                }
                .controlSize(.small)
            }
            .padding(14)
            .background(.red.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))

        case .ready:
            EmptyView()
        }
    }

    private var notDownloadedBanner: some View {
        HStack(spacing: 14) {
            Image(systemName: "arrow.down.circle.fill")
                .font(.title2)
                .foregroundStyle(.tint)

            VStack(alignment: .leading, spacing: 3) {
                Text("No model downloaded")
                    .font(.callout.bold())
                Text("Download \(settings.ditVariant.displayName) to start generating.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            Button("Download") {
                downloadSheetVariant = settings.ditVariant
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
        }
        .padding(14)
        .background(.tint.opacity(0.07), in: RoundedRectangle(cornerRadius: 10))
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(.tint.opacity(0.2), lineWidth: 1)
        )
    }
}

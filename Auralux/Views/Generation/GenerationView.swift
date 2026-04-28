import SwiftUI

struct GenerationView: View {
    @Environment(GenerationViewModel.self) private var viewModel
    @Environment(SettingsViewModel.self) private var settings
    @Environment(NativeInferenceEngine.self) private var engine
    @Environment(\.modelContext) private var modelContext
    @State private var tagText = ""
    @State private var completionDismissTask: Task<Void, Never>?
    @State private var didSeedDefaults = false

    private var engineReady: Bool {
        engine.modelState.isReady
    }

    private var generateDisabled: Bool {
        viewModel.state.isBusy || !engine.modelState.isReady || engine.isGenerating
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if !engineReady && !engine.isOnboarding {
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

                HStack {
                    Button("Generate") {
                        viewModel.generate(in: modelContext)
                    }
                    .keyboardShortcut("g", modifiers: [.command])
                    .disabled(generateDisabled)
                    .accessibilityIdentifier("generate-button")

                    Button("Cancel") {
                        viewModel.cancel()
                    }
                    .disabled(!viewModel.state.isBusy)
                    .accessibilityIdentifier("cancel-button")

                    Spacer()

                    switch viewModel.state {
                    case .generating, .preparing:
                        VStack(alignment: .trailing, spacing: 4) {
                            ProgressView(value: viewModel.progress)
                                .frame(width: 200)
                                .accessibilityIdentifier("generation-progress")
                            if !viewModel.progressMessage.isEmpty {
                                Text(viewModel.progressMessage)
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(1)
                            }
                        }
                    case .completed:
                        Label("Done", systemImage: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                            .accessibilityIdentifier("generation-status")
                            .onAppear {
                                completionDismissTask?.cancel()
                                completionDismissTask = Task {
                                    try? await Task.sleep(for: .seconds(3))
                                    if !Task.isCancelled {
                                        viewModel.state = .idle
                                    }
                                }
                            }
                            .onDisappear {
                                completionDismissTask?.cancel()
                            }
                    case .failed(let message):
                        HStack(spacing: 6) {
                            Label(message, systemImage: "exclamationmark.triangle.fill")
                                .foregroundStyle(.red)
                                .accessibilityIdentifier("generation-status")
                            Button {
                                viewModel.state = .idle
                            } label: {
                                Image(systemName: "xmark.circle.fill")
                                    .foregroundStyle(.secondary)
                            }
                            .buttonStyle(.plain)
                            .accessibilityLabel("Dismiss error")
                        }
                    case .idle:
                        EmptyView()
                    }
                }
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
    }

    @ViewBuilder
    private var modelBanner: some View {
        HStack(spacing: 12) {
            Image(systemName: bannerIcon)
                .foregroundStyle(bannerColor)

            VStack(alignment: .leading, spacing: 2) {
                Text(bannerTitle)
                    .font(.callout.bold())
                Text(bannerMessage)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            if engine.modelState.isLoading {
                ProgressView()
                    .controlSize(.small)
            } else if case .error = engine.modelState {
                Button("Retry") {
                    Task { await engine.loadModels() }
                }
                .controlSize(.small)
            }
        }
        .padding(12)
        .background(.orange.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
    }

    private var bannerIcon: String {
        switch engine.modelState {
        case .error: return "exclamationmark.triangle.fill"
        case .loading: return "arrow.triangle.2.circlepath"
        default: return "info.circle"
        }
    }

    private var bannerColor: Color {
        switch engine.modelState {
        case .error: return .red
        case .loading: return .orange
        default: return .yellow
        }
    }

    private var bannerTitle: String {
        switch engine.modelState {
        case .notDownloaded:            return "Models Not Downloaded"
        case .downloading:              return "Downloading Models…"
        case .loading:                  return "Loading Models…"
        case .error:                    return "Model Load Error"
        case .ready:                    return "Ready"
        }
    }

    private var bannerMessage: String {
        switch engine.modelState {
        case .notDownloaded:
            return "Open setup to download model weights automatically."
        case .downloading(let progress):
            return "Downloading model weights: \(Int(progress * 100))% complete."
        case .loading:
            return "Loading model weights into memory. This may take a moment."
        case .error(let message):
            return message
        case .ready:
            return ""
        }
    }
}

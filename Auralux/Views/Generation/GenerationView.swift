import SwiftUI

struct GenerationView: View {
    @Environment(GenerationViewModel.self) private var viewModel
    @Environment(EngineService.self) private var engine
    @Environment(\.modelContext) private var modelContext
    @State private var tagText = ""
    @State private var completionDismissTask: Task<Void, Never>?

    private var engineReady: Bool {
        engine.state.isReady
    }

    private var generateDisabled: Bool {
        viewModel.state.isBusy || engine.state.isBusy || engine.isControlActionRunning
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if !engineReady && !engine.isOnboarding {
                    engineBanner
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
                ParameterControlsView(duration: Bindable(viewModel).duration, variance: Bindable(viewModel).variance, seedText: Bindable(viewModel).seedText)

                HStack {
                    Button("Generate") {
                        viewModel.generate(in: modelContext, engine: engine)
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
    }

    @ViewBuilder
    private var engineBanner: some View {
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

            if engine.state.isBusy {
                ProgressView()
                    .controlSize(.small)
            } else if engine.state.needsSetup {
                Button("Set Up") {
                    Task { await engine.runSetup() }
                }
                .controlSize(.small)
            } else if engine.state.isStopped {
                Button("Start Now") {
                    Task { await engine.startServer() }
                }
                .controlSize(.small)
            } else if case .error = engine.state {
                Button("Retry") {
                    Task { await engine.startServer() }
                }
                .controlSize(.small)
            }
        }
        .padding(12)
        .background(.orange.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
    }

    private var bannerIcon: String {
        switch engine.state {
        case .error: return "exclamationmark.triangle.fill"
        case .settingUp, .starting: return "arrow.triangle.2.circlepath"
        default: return "info.circle"
        }
    }

    private var bannerColor: Color {
        switch engine.state {
        case .error: return .red
        case .settingUp, .starting: return .orange
        default: return .yellow
        }
    }

    private var bannerTitle: String {
        switch engine.state {
        case .notSetup: return "Engine Not Configured"
        case .settingUp: return "Setting Up..."
        case .stopped: return "Engine Idle"
        case .starting: return "Server Starting..."
        case .error: return "Engine Error"
        default: return "Engine Not Ready"
        }
    }

    private var bannerMessage: String {
        switch engine.state {
        case .notSetup:
            return "Set up the inference engine to start generating music."
        case .settingUp(let progress):
            return progress
        case .stopped:
            return "The server stays offline until you generate audio. Starting generation will launch it automatically."
        case .starting:
            return "The inference server is starting up. This may take a moment."
        case .error(let message):
            return message
        default:
            return "The inference engine is not yet available."
        }
    }
}

import SwiftUI

/// A compact status badge showing the current state of the MLX inference models.
/// Designed for use in the toolbar.
struct EngineStatusView: View {
    @Environment(NativeInferenceEngine.self) private var engine

    @State private var showPopover = false

    var body: some View {
        Button {
            showPopover.toggle()
        } label: {
            HStack(spacing: 5) {
                Circle()
                    .fill(statusColor)
                    .frame(width: 7, height: 7)

                Text(statusLabel)
                    .font(.caption)
                    .lineLimit(1)
                    .fixedSize()
            }
            .padding(.horizontal, 4)
            .padding(.vertical, 2)
        }
        .buttonStyle(.glass)
        .fixedSize()
        .accessibilityLabel("Model status: \(statusLabel)")
        .popover(isPresented: $showPopover) {
            modelPopover
                .padding(16)
                .frame(width: 280)
        }
        .help("Model status — click for details")
    }

    private var statusColor: Color {
        switch engine.modelState {
        case .ready:
            return engine.isGenerating ? .yellow : .green
        case .downloaded:
            return engine.isGenerating ? .yellow : .green.opacity(0.6)
        case .loading:
            return .orange
        case .downloading:
            return .blue
        case .notDownloaded:
            return .gray
        case .error:
            return .red
        }
    }

    private var statusLabel: String {
        switch engine.modelState {
        case .ready:
            return engine.isGenerating ? "Generating" : "Ready"
        case .downloaded:
            return engine.isGenerating ? "Generating" : "Ready"
        case .loading:
            return "Loading"
        case .downloading(let p):
            return "Downloading \(Int(p * 100))%"
        case .notDownloaded:
            return "Not downloaded"
        case .error:
            return "Error"
        }
    }

    @ViewBuilder
    private var modelPopover: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Circle()
                    .fill(statusColor)
                    .frame(width: 9, height: 9)

                VStack(alignment: .leading, spacing: 2) {
                    Text("MLX Inference")
                        .font(.subheadline.weight(.semibold))
                    Text(statusLabel)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            switch engine.modelState {
            case .ready:
                Divider()
                Label("Models loaded and ready", systemImage: "checkmark.circle.fill")
                    .font(.callout)
                    .foregroundStyle(.green)
                if engine.isGenerating {
                    Label("Generation in progress...", systemImage: "waveform")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

            case .downloaded:
                Divider()
                Label("Models load automatically on first generate", systemImage: "arrow.clockwise")
                    .font(.callout)
                    .foregroundStyle(.secondary)

            case .loading:
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Loading model weights...")
                        .font(.callout)
                }

            case .downloading(let progress):
                VStack(alignment: .leading, spacing: 6) {
                    Label("Downloading model weights…", systemImage: "arrow.down.circle")
                        .font(.callout)
                        .foregroundStyle(.blue)
                    ProgressView(value: progress)
                        .progressViewStyle(.linear)
                    Text("\(Int(progress * 100))% — ~5 GB total")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }

            case .notDownloaded:
                VStack(alignment: .leading, spacing: 6) {
                    Label("Weights not found", systemImage: "exclamationmark.triangle")
                        .font(.callout)
                        .foregroundStyle(.orange)
                    Text("Open Auralux and let it download model weights automatically.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

            case .error(let message):
                VStack(alignment: .leading, spacing: 6) {
                    Label("Load error", systemImage: "xmark.circle.fill")
                        .font(.callout)
                        .foregroundStyle(.red)
                    Text(message)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(3)
                    Button("Retry") {
                        Task { await engine.loadModels() }
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
            }
        }
    }
}

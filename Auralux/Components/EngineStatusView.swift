import SwiftUI

/// A compact status badge showing the current state of the inference engine.
/// Designed for use in the toolbar or sidebar.
struct EngineStatusView: View {
    @Environment(EngineService.self) private var engine

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
        .accessibilityLabel("Engine status: \(statusLabel)")
        .popover(isPresented: $showPopover) {
            enginePopover
                .padding(16)
                .frame(width: 320)
        }
        .help("Engine status — click for details")
    }

    private var statusColor: Color {
        switch engine.state {
        case .ready:
            return .green
        case .running:
            return .yellow
        case .starting, .settingUp:
            return .orange
        case .error:
            return .red
        case .notSetup, .unknown, .stopped:
            return .gray
        }
    }

    private var statusLabel: String {
        switch engine.state {
        case .ready:
            return "Ready"
        case .running:
            return "Running"
        case .starting:
            return "Starting"
        case .settingUp:
            return "Setting up"
        case .error:
            return "Error"
        case .notSetup:
            return "Not configured"
        case .unknown:
            return "Checking"
        case .stopped:
            return "Idle"
        }
    }

    @ViewBuilder
    private var enginePopover: some View {
        VStack(alignment: .leading, spacing: 12) {
            EngineControlPanel(isCompact: true)

            switch engine.state {
            case .ready:
                Divider()
                readyInfo
            case .running:
                Label("Server running, models loading...", systemImage: "bolt.circle")
                    .font(.callout)
            case .starting:
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Starting server...")
                        .font(.callout)
                }
            case .settingUp(let progress):
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text(progress)
                        .font(.callout)
                        .lineLimit(2)
                }
            case .error:
                Button("Retry") {
                    Task { await engine.startServer() }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

            case .notSetup:
                Label("Engine not configured. Open Settings to set up.", systemImage: "gearshape")
                    .font(.callout)
                    .foregroundStyle(.secondary)

            case .unknown:
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Checking engine status...")
                        .font(.callout)
                }
            case .stopped:
                Label("Server idle. It will start on demand when you generate audio.", systemImage: "pause.circle")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder
    private var readyInfo: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Engine ready", systemImage: "checkmark.circle.fill")
                .font(.callout)
                .foregroundStyle(.green)

            if engine.modelStatus.ditLoaded {
                HStack(spacing: 12) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Device").font(.caption2).foregroundStyle(.tertiary)
                        Text(engine.modelStatus.device).font(.caption)
                    }
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Engine").font(.caption2).foregroundStyle(.tertiary)
                        Text(engine.modelStatus.engine).font(.caption)
                    }
                }

                if !engine.modelStatus.ditModel.isEmpty {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("DiT Model").font(.caption2).foregroundStyle(.tertiary)
                        Text(engine.modelStatus.ditModel).font(.caption)
                    }
                }

                if engine.modelStatus.llmLoaded, !engine.modelStatus.llmModel.isEmpty {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("LM Model").font(.caption2).foregroundStyle(.tertiary)
                        Text(engine.modelStatus.llmModel).font(.caption)
                    }
                }
            }
        }
    }
}

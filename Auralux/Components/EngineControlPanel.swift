import SwiftUI

struct EngineControlPanel: View {
    @Environment(EngineService.self) private var engine

    var isCompact = false

    var body: some View {
        VStack(alignment: .leading, spacing: isCompact ? 10 : 14) {
            header
            controls

            if engine.state.isRunning || engine.runtimeStats.pid != nil {
                Divider()
                statsSection
            }

            if case .error(let message) = engine.state {
                Label(message, systemImage: "exclamationmark.triangle.fill")
                    .font(.callout)
                    .foregroundStyle(.red)
            }
        }
    }

    private var header: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(statusColor)
                .frame(width: 9, height: 9)

            VStack(alignment: .leading, spacing: 2) {
                Text("Inference Engine")
                    .font(isCompact ? .subheadline.weight(.semibold) : .headline)
                Text(statusLabel)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            Button {
                Task { await engine.refreshNow() }
            } label: {
                Image(systemName: "arrow.clockwise")
            }
            .buttonStyle(.borderless)
            .disabled(engine.isControlActionRunning)
            .opacity(engine.isControlActionRunning ? 0.5 : 1)
            .help("Refresh engine status")
            .accessibilityLabel("Refresh engine status")
        }
    }

    private var controls: some View {
        HStack(spacing: 8) {
            Button {
                Task { await engine.startServer() }
            } label: {
                Label("Start", systemImage: "play.fill")
            }
            .disabled(!canStart)

            Button {
                engine.stopServer()
            } label: {
                Label("Stop", systemImage: "stop.fill")
            }
            .disabled(!canStop)

            Button {
                Task { await engine.restartServer() }
            } label: {
                Label("Restart", systemImage: "arrow.triangle.2.circlepath")
            }
            .disabled(!canRestart)

            if engine.isControlActionRunning || engine.state.isBusy {
                ProgressView()
                    .controlSize(.small)
            }
        }
        .buttonStyle(.bordered)
        .controlSize(.small)
        .opacity(engine.isControlActionRunning ? 0.5 : 1)
    }

    private var statsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            statRow("PID", value: engine.runtimeStats.pid.map(String.init) ?? "n/a")
            statRow("Uptime", value: formattedUptime)
            statRow("CPU", value: engine.runtimeStats.cpuPercent.map { String(format: "%.1f%%", $0) } ?? "n/a")
            statRow("Memory", value: engine.runtimeStats.memoryRSSMB.map { String(format: "%.1f MB", $0) } ?? "n/a")
            statRow("Threads", value: engine.runtimeStats.activeThreads.map(String.init) ?? "n/a")
            statRow("Jobs", value: formattedJobs)
            statRow("Health", value: formattedHealth)

            if let statsError = engine.runtimeStats.statsError {
                Text(statsError)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .lineLimit(2)
            }
        }
        .font(.caption)
    }

    private func statRow(_ title: String, value: String) -> some View {
        HStack {
            Text(title)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .monospacedDigit()
                .lineLimit(1)
        }
    }

    private var canStart: Bool {
        !engine.state.isBusy
        && !engine.state.isRunning
        && !engine.isControlActionRunning
        && !engine.state.needsSetup
    }

    private var canStop: Bool {
        (engine.state.isRunning || engine.state == .starting || engine.isManagedServerRunning || engine.runtimeStats.pid != nil)
        && !engine.isControlActionRunning
    }

    private var canRestart: Bool {
        !engine.state.isBusy
        && !engine.isControlActionRunning
        && !engine.state.needsSetup
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
            return "Server running, models loading"
        case .starting:
            return "Starting"
        case .settingUp(let progress):
            return progress
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

    private var formattedUptime: String {
        guard let uptime = engine.runtimeStats.uptimeSeconds else { return "n/a" }
        let seconds = Int(uptime)
        let hours = seconds / 3600
        let minutes = (seconds % 3600) / 60
        let remainder = seconds % 60
        if hours > 0 {
            return "\(hours)h \(minutes)m"
        }
        if minutes > 0 {
            return "\(minutes)m \(remainder)s"
        }
        return "\(remainder)s"
    }

    private var formattedJobs: String {
        guard let jobCounts = engine.runtimeStats.jobCounts, !jobCounts.isEmpty else { return "none" }
        return jobCounts
            .sorted { $0.key < $1.key }
            .map { "\($0.key): \($0.value)" }
            .joined(separator: ", ")
    }

    private var formattedHealth: String {
        guard let checkedAt = engine.lastHealthCheckAt else { return "not checked" }
        let time = checkedAt.formatted(date: .omitted, time: .standard)
        if let latency = engine.lastHealthLatency {
            return "\(time), \(Int(latency * 1000)) ms"
        }
        return time
    }
}

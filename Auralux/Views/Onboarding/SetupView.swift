import SwiftUI

/// Compact glass overlay that provisions the local engine environment.
struct SetupView: View {
    @Environment(EngineService.self) private var engine

    @State private var stepStatuses: [Step: StepStatus] = Step.allCases.reduce(into: [:]) { $0[$1] = .pending }
    @State private var activeStep: Step?
    @State private var detailText = ""
    @State private var hasError = false
    @State private var setupTask: Task<Void, Never>?

    var body: some View {
        VStack(spacing: 0) {
            header
                .padding(.top, 24)
                .padding(.bottom, 16)

            Divider().opacity(0.3)

            stepList
                .padding(.horizontal, 28)
                .padding(.vertical, 20)

            if !detailText.isEmpty {
                Text(detailText)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: .infinity)
                    .padding(.horizontal, 28)
                    .padding(.bottom, 16)
                    .transition(.opacity)
            }

            Divider().opacity(0.3)

            footer
                .padding(.horizontal, 24)
                .padding(.vertical, 12)
        }
        .frame(width: 380)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
        .shadow(color: .black.opacity(0.15), radius: 24, y: 8)
        .task {
            setupTask = Task { await runAllSteps() }
            await setupTask?.value
        }
        .onDisappear {
            setupTask?.cancel()
        }
        .onChange(of: engine.state) { _, newState in
            updateDetailText(for: newState)
        }
    }

    // MARK: - Subviews

    private var header: some View {
        VStack(spacing: 6) {
            Image(systemName: "wand.and.stars")
                .font(.system(size: 32))
                .foregroundStyle(.tint)

            Text("Setting Up Auralux")
                .font(.headline)

            Text("AI Music Generation on Apple Silicon")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var stepList: some View {
        VStack(alignment: .leading, spacing: 14) {
            ForEach(Step.allCases, id: \.rawValue) { step in
                stepRow(step)
            }
        }
    }

    private func stepRow(_ step: Step) -> some View {
        let status = stepStatuses[step] ?? .pending

        return HStack(spacing: 12) {
            statusIcon(for: status)
                .frame(width: 20, height: 20)

            Text(step.label)
                .font(.callout)
                .foregroundStyle(status == .pending ? .secondary : .primary)

            Spacer()

            if status == .completed {
                Image(systemName: "checkmark")
                    .font(.caption.bold())
                    .foregroundStyle(.green)
            }
        }
    }

    @ViewBuilder
    private func statusIcon(for status: StepStatus) -> some View {
        switch status {
        case .pending:
            Circle()
                .stroke(Color.secondary.opacity(0.3), lineWidth: 2)
        case .inProgress:
            ProgressView()
                .controlSize(.small)
        case .completed:
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(.green)
        case .error:
            Image(systemName: "xmark.circle.fill")
                .foregroundStyle(.red)
        }
    }

    private var footer: some View {
        HStack {
            Button("Cancel") {
                withAnimation { engine.isOnboarding = false }
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
            .font(.callout)

            Spacer()

            if hasError {
                Button("Retry") {
                    retryFromError()
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
            }
        }
    }

    // MARK: - Auto-advance Logic

    private func runAllSteps() async {
        hasError = false

        // Step 1: System Check
        if stepStatuses[.systemCheck] != .completed {
            await runStep(.systemCheck) {
                let hasEngine = engine.engineDirectory != nil

                if !hasEngine {
                    throw SetupError.message("AuraluxEngine directory not found.")
                }
                #if !arch(arm64)
                    throw SetupError.message("Apple Silicon (M1 or later) is required.")
                #endif
            }
            guard !hasError, !Task.isCancelled else { return }
        }

        // Step 2: Environment Setup
        if stepStatuses[.environmentSetup] != .completed {
            if engine.isACEStepCloned && engine.isVenvReady {
                withAnimation { stepStatuses[.environmentSetup] = .completed }
            } else {
                await runStep(.environmentSetup) {
                    await engine.runSetup()
                    if !engine.isVenvReady {
                        if case .error(let msg) = engine.state {
                            throw SetupError.message(msg)
                        }
                        throw SetupError.message("Environment setup did not complete successfully.")
                    }
                }
                guard !hasError, !Task.isCancelled else { return }
            }
        }

        // All done -- brief pause then auto-dismiss
        withAnimation { detailText = "" }
        try? await Task.sleep(for: .milliseconds(750))
        if !Task.isCancelled {
            withAnimation { engine.isOnboarding = false }
        }
    }

    private func runStep(_ step: Step, action: () async throws -> Void) async {
        withAnimation {
            activeStep = step
            stepStatuses[step] = .inProgress
            detailText = ""
        }

        do {
            try await action()
            if !Task.isCancelled {
                withAnimation { stepStatuses[step] = .completed }
            }
        } catch {
            if !Task.isCancelled {
                withAnimation {
                    stepStatuses[step] = .error(error.localizedDescription)
                    detailText = error.localizedDescription
                }
                hasError = true
            }
        }
    }

    private func retryFromError() {
        var shouldReset = false
        for step in Step.allCases {
            if case .error = stepStatuses[step] {
                shouldReset = true
            }
            if shouldReset {
                stepStatuses[step] = .pending
            }
        }
        hasError = false
        detailText = ""
        setupTask?.cancel()
        setupTask = Task { await runAllSteps() }
    }

    private func updateDetailText(for newState: EngineState) {
        guard let step = activeStep else { return }
        switch step {
        case .environmentSetup:
            if case .settingUp(let progress) = newState {
                withAnimation { detailText = progress }
            }
        default:
            break
        }
    }
}

// MARK: - Supporting Types

extension SetupView {
    enum Step: Int, CaseIterable {
        case systemCheck
        case environmentSetup

        var label: String {
            switch self {
            case .systemCheck: return "System Check"
            case .environmentSetup: return "Environment Setup"
            }
        }
    }

    enum StepStatus: Equatable {
        case pending
        case inProgress
        case completed
        case error(String)

        var isError: Bool {
            if case .error = self { return true }
            return false
        }
    }

    private enum SetupError: LocalizedError {
        case message(String)

        var errorDescription: String? {
            switch self {
            case .message(let msg): return msg
            }
        }
    }
}

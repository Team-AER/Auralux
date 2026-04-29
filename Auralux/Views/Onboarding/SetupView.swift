import SwiftUI

/// Compact glass overlay that provisions the local inference environment.
struct SetupView: View {
    @Environment(NativeInferenceEngine.self) private var engine

    @State private var stepStatuses: [Step: StepStatus] = Step.allCases.reduce(into: [:]) { $0[$1] = .pending }
    @State private var activeStep: Step?
    @State private var detailText = ""
    @State private var hasError = false
    @State private var setupTask: Task<Void, Never>?

    private var downloadProgress: Double? {
        if case .downloading(let p) = engine.modelState { return p }
        return nil
    }

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
        .frame(width: 420)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
        .shadow(color: .black.opacity(0.15), radius: 24, y: 8)
        .task {
            setupTask = Task { await runAllSteps() }
            await setupTask?.value
        }
        .onDisappear {
            setupTask?.cancel()
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
        let isDownloading = step == .modelWeights && downloadProgress != nil

        return VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 12) {
                statusIcon(for: status)
                    .frame(width: 20, height: 20)

                VStack(alignment: .leading, spacing: 1) {
                    Text(step.label)
                        .font(.callout)
                        .foregroundStyle(status == .pending ? .secondary : .primary)

                    if isDownloading, let p = downloadProgress {
                        Text("\(Int(p * 100))% — downloading model weights (~5.4 GB)")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }

                Spacer()

                if status == .completed {
                    Image(systemName: "checkmark")
                        .font(.caption.bold())
                        .foregroundStyle(.green)
                }
            }

            if isDownloading, let p = downloadProgress {
                ProgressView(value: p)
                    .progressViewStyle(.linear)
                    .padding(.leading, 32)
                    .transition(.opacity)
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
                #if !arch(arm64)
                throw SetupError.message("Apple Silicon (M1 or later) is required.")
                #endif
            }
            guard !hasError, !Task.isCancelled else { return }
        }

        // Step 2: Model Weights — download from HuggingFace if missing; loading happens on first generate
        if stepStatuses[.modelWeights] != .completed {
            await runStep(.modelWeights) {
                if !engine.weightsExist {
                    try await engine.downloadAndLoad()
                }
                if case .error(let msg) = engine.modelState {
                    throw SetupError.message(msg)
                }
            }
            guard !hasError, !Task.isCancelled else { return }
        }

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
            if case .error = stepStatuses[step] { shouldReset = true }
            if shouldReset { stepStatuses[step] = .pending }
        }
        hasError = false
        detailText = ""
        setupTask?.cancel()
        setupTask = Task { await runAllSteps() }
    }
}

// MARK: - Supporting Types

extension SetupView {
    enum Step: Int, CaseIterable {
        case systemCheck
        case modelWeights

        var label: String {
            switch self {
            case .systemCheck: return "System Check"
            case .modelWeights: return "Model Weights"
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

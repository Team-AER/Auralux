import AppKit
import SwiftData
import SwiftUI

struct ContentView: View {
    @Environment(SidebarViewModel.self) private var sidebarViewModel
    @Environment(HistoryViewModel.self) private var historyViewModel
    @Environment(GenerationViewModel.self) private var generationViewModel
    @Environment(NativeInferenceEngine.self) private var engine
    @Environment(\.modelContext) private var modelContext
    @State private var didBootstrap = false
    @State private var showLowRAMWarning = false

    @State private var playerPanelWidth: CGFloat? = nil
    @State private var dragStartWidth: CGFloat = 0
    @State private var completionDismissTask: Task<Void, Never>?
    private let minPanelWidth: CGFloat = 280

    private static let lowRAMWarningDismissedKey = "warning.lowRAMDismissed"

    private var isGenerateSection: Bool {
        (sidebarViewModel.selectedSection ?? .generate) == .generate
    }

    private var primaryActionDisabled: Bool {
        if generationViewModel.state.isBusy { return false }
        return !engine.modelState.isReady || engine.isGenerating
    }

    private var primaryActionIcon: String {
        generationViewModel.state.isBusy ? "stop.fill" : "sparkles"
    }

    private var primaryActionLabel: String {
        generationViewModel.state.isBusy ? "Cancel" : "Generate"
    }

    private var primaryActionTint: Color {
        generationViewModel.state.isBusy ? .red : .accentColor
    }

    var body: some View {
        ZStack {
            mainContent

            if engine.isOnboarding {
                Color.black.opacity(0.3)
                    .ignoresSafeArea()
                    .transition(.opacity)

                SetupView()
                    .transition(.opacity.combined(with: .scale(scale: 0.95)))
            }
        }
        .task {
            guard !didBootstrap else { return }
            didBootstrap = true

            let presetService = PresetService(context: modelContext)
            try? presetService.bootstrapFromBundleIfNeeded()
            try? await HistoryService(context: modelContext).reconcileOrphans()
            historyViewModel.refresh(context: modelContext)

            await engine.checkStatus()

            if AppConstants.isUnderspecMachine,
               !UserDefaults.standard.bool(forKey: Self.lowRAMWarningDismissedKey) {
                showLowRAMWarning = true
            }
        }
        .alert("Low memory device", isPresented: $showLowRAMWarning) {
            Button("OK") {
                UserDefaults.standard.set(true, forKey: Self.lowRAMWarningDismissedKey)
            }
        } message: {
            Text("Auralux works best on Macs with 32 GB or more RAM. On this machine generation may swap heavily or fail under load. Low-memory mode is enabled by default — you can adjust it in Settings.")
        }
    }

    @ViewBuilder
    private var playerPanel: some View {
        if let selectedTrack = historyViewModel.selectedTrack ?? generationViewModel.lastTrack {
            PlayerView(track: selectedTrack)
        } else {
            ContentUnavailableView("No Track Selected", systemImage: "music.note", description: Text("Generate or select a track to preview it."))
        }
    }

    private func resolvedPlayerWidth(totalWidth: CGFloat) -> CGFloat {
        let target = playerPanelWidth ?? (totalWidth / 2)
        let maxWidth = max(minPanelWidth, totalWidth - minPanelWidth)
        return max(minPanelWidth, min(maxWidth, target))
    }

    private func resizeDivider(totalWidth: CGFloat) -> some View {
        ZStack {
            Rectangle()
                .fill(Color(nsColor: .separatorColor))
                .frame(width: 1)
            Color.clear
                .frame(width: 8)
                .contentShape(Rectangle())
                .onHover { inside in
                    if inside { NSCursor.resizeLeftRight.push() } else { NSCursor.pop() }
                }
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { value in
                            let proposed = dragStartWidth - value.translation.width
                            let maxWidth = max(minPanelWidth, totalWidth - minPanelWidth)
                            playerPanelWidth = max(minPanelWidth, min(maxWidth, proposed))
                        }
                        .onEnded { _ in dragStartWidth = playerPanelWidth ?? 0 }
                )
        }
        .frame(width: 8)
    }

    private var mainContent: some View {
        NavigationSplitView {
            SidebarView()
        } detail: {
            GeometryReader { proxy in
                HStack(spacing: 0) {
                    Group {
                        switch sidebarViewModel.selectedSection ?? .generate {
                        case .generate:
                            GenerationView()
                        case .history:
                            HistoryBrowserView()
                        case .audioToAudio:
                            AudioImportView()
                        case .settings:
                            SettingsView()
                        }
                    }
                    .navigationTitle(sidebarViewModel.selectedSection?.title ?? "Auralux")
                    .frame(maxWidth: .infinity)

                    if sidebarViewModel.selectedSection != .settings {
                        resizeDivider(totalWidth: proxy.size.width)
                        playerPanel
                            .frame(width: resolvedPlayerWidth(totalWidth: proxy.size.width))
                    }
                }
                .onAppear {
                    if playerPanelWidth == nil {
                        let half = proxy.size.width / 2
                        playerPanelWidth = half
                        dragStartWidth = half
                    }
                }
            }
        }
        .toolbar {
            if isGenerateSection {
                ToolbarItem(placement: .automatic) {
                    Button {
                        if generationViewModel.state.isBusy {
                            generationViewModel.cancel()
                        } else {
                            generationViewModel.generate(in: modelContext)
                        }
                    } label: {
                        HStack(spacing: 6) {
                            Image(systemName: primaryActionIcon)
                                .contentTransition(.symbolEffect(.replace.downUp))
                                .symbolEffect(
                                    .pulse,
                                    options: .repeat(.continuous),
                                    isActive: generationViewModel.state.isBusy
                                )
                            Text(primaryActionLabel)
                        }
                    }
                    .buttonStyle(.glassProminent)
                    .tint(primaryActionTint)
                    .keyboardShortcut("g", modifiers: [.command])
                    .disabled(primaryActionDisabled)
                    .animation(.smooth(duration: 0.3), value: generationViewModel.state.isBusy)
                    .accessibilityIdentifier("generate-cancel-button")
                }

                ToolbarSpacer(.fixed)

                ToolbarItem(placement: .automatic) {
                    generationStatusView
                }
            }

            ToolbarSpacer(.fixed)

            ToolbarItem(placement: .automatic) {
                EngineStatusView()
            }
        }
    }

    @ViewBuilder
    private var generationStatusView: some View {
        switch generationViewModel.state {
        case .preparing, .generating:
            if !generationViewModel.progressMessage.isEmpty {
                Text(generationViewModel.progressMessage)
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .fixedSize()
                    .padding(.horizontal, 14)
                    .accessibilityIdentifier("generation-progress-counter")
            }
        case .completed:
            Label("Done", systemImage: "checkmark.circle.fill")
                .font(.caption)
                .foregroundStyle(.green)
                .labelStyle(.titleAndIcon)
                .padding(.horizontal, 14)
                .accessibilityIdentifier("generation-status")
                .onAppear {
                    completionDismissTask?.cancel()
                    completionDismissTask = Task {
                        try? await Task.sleep(for: .seconds(3))
                        if !Task.isCancelled {
                            generationViewModel.state = .idle
                        }
                    }
                }
                .onDisappear { completionDismissTask?.cancel() }
        case .failed(let message):
            HStack(spacing: 4) {
                Label(message, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.red)
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .help(message)
                    .accessibilityIdentifier("generation-status")
                Button {
                    generationViewModel.state = .idle
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Dismiss error")
            }
            .padding(.horizontal, 14)
        case .idle:
            EmptyView()
        }
    }
}

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

    @State private var playerPanelWidth: CGFloat? = nil
    @State private var dragStartWidth: CGFloat = 0
    private let minPanelWidth: CGFloat = 280

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
            ToolbarSpacer(.fixed)

            ToolbarItem(placement: .automatic) {
                EngineStatusView()
            }
        }
    }
}

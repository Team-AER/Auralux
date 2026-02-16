import SwiftData
import SwiftUI

struct ContentView: View {
    @Environment(SidebarViewModel.self) private var sidebarViewModel
    @Environment(HistoryViewModel.self) private var historyViewModel
    @Environment(GenerationViewModel.self) private var generationViewModel
    @Environment(EngineService.self) private var engineService
    @Environment(\.modelContext) private var modelContext
    @Environment(\.openWindow) private var openWindow
    @State private var didBootstrap = false

    var body: some View {
        ZStack {
            mainContent

            if engineService.isOnboarding {
                SetupView()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(.background)
                    .transition(.opacity)
            }
        }
        .task {
            guard !didBootstrap else { return }
            didBootstrap = true

            let presetService = PresetService(context: modelContext)
            try? presetService.bootstrapFromBundleIfNeeded()
            historyViewModel.refresh(context: modelContext)

            // Check engine status on launch
            await engineService.checkStatus()

            if !engineService.state.isReady && !engineService.state.isRunning {
                withAnimation { engineService.isOnboarding = true }
            }
        }
        .onChange(of: engineService.state) { _, newState in
            if newState.isReady {
                withAnimation { engineService.isOnboarding = false }
            }
        }
    }

    private var mainContent: some View {
        NavigationSplitView {
            SidebarView()
        } content: {
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
        } detail: {
            if let selectedTrack = historyViewModel.selectedTrack ?? generationViewModel.lastTrack {
                PlayerView(track: selectedTrack)
            } else {
                ContentUnavailableView("No Track Selected", systemImage: "music.note", description: Text("Generate or select a track to preview it."))
            }
        }
        .toolbar {
            ToolbarItem(placement: .automatic) {
                HStack(spacing: 12) {
                    Button {
                        openWindow(id: "log-viewer")
                    } label: {
                        Image(systemName: "terminal")
                            .help("Show Logs (Cmd+Opt+L)")
                    }
                    .keyboardShortcut("l", modifiers: [.command, .option])

                    EngineStatusView()

                    Text("Auralux")
                        .font(.headline)
                }
            }
        }
    }
}

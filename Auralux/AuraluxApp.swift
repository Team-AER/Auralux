import AppKit
import SwiftData
import SwiftUI

/// Ensures the SPM-built executable is promoted to a regular GUI application
/// so macOS gives it a menu bar, Dock icon, and keyboard focus.
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    var onTerminate: (() -> Void)?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    func applicationWillTerminate(_ notification: Notification) {
        onTerminate?()
    }
}

@main
struct AuraluxApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    @State private var sidebarViewModel = SidebarViewModel()
    @State private var generationViewModel: GenerationViewModel
    @State private var historyViewModel = HistoryViewModel()
    @State private var playerViewModel = PlayerViewModel()
    @State private var settingsViewModel = SettingsViewModel()
    @State private var engineService: EngineService

    private let modelContainer: ModelContainer

    init() {
        let inferenceService = InferenceService()
        _generationViewModel = State(initialValue: GenerationViewModel(inferenceService: inferenceService))
        _engineService = State(initialValue: EngineService(inferenceService: inferenceService))

        do {
            modelContainer = try ModelContainer(
                for: GeneratedTrack.self,
                Preset.self,
                Tag.self
            )
        } catch {
            fatalError("Failed to initialize SwiftData container: \(error)")
        }

        AppLogger.shared.info("Auralux launched", category: .app)
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(minWidth: AppConstants.minimumWindowWidth, minHeight: AppConstants.minimumWindowHeight)
                .environment(sidebarViewModel)
                .environment(generationViewModel)
                .environment(historyViewModel)
                .environment(playerViewModel)
                .environment(settingsViewModel)
                .environment(engineService)
                .onAppear {
                    appDelegate.onTerminate = {
                        playerViewModel.playerService.shutdown()
                        engineService.shutdown()
                    }
                }
        }
        .modelContainer(modelContainer)
        .defaultSize(width: 1280, height: 840)
        .commands {
            CommandGroup(replacing: .appInfo) {
                Button("About Auralux") {
                    NSApplication.shared.orderFrontStandardAboutPanel(options: [
                        NSApplication.AboutPanelOptionKey.applicationName: AppConstants.appName,
                        NSApplication.AboutPanelOptionKey.applicationVersion: "1.0.0",
                    ])
                }
            }
            CommandGroup(replacing: .newItem) {}
        }

        Window("Auralux Logs", id: "log-viewer") {
            LogViewerView()
        }
        .defaultSize(width: 800, height: 500)
    }
}

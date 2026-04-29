import AppKit
import MLX
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
    @State private var engine: NativeInferenceEngine

    private let modelContainer: ModelContainer

    init() {
        // Cap the MLX freed-buffer pool. By default MLX retains every buffer it has
        // allocated for reuse, which makes resident memory grow to the high-water mark
        // of the union of every phase (weight load + DiT activations + VAE decode) and
        // never shrink. Halve it under low-memory mode.
        let lowMemory = UserDefaults.standard.bool(forKey: "settings.lowMemoryMode")
        MLX.Memory.cacheLimit = (lowMemory ? 512 : 1024) * 1024 * 1024

        let engine = NativeInferenceEngine()
        _engine = State(initialValue: engine)
        _generationViewModel = State(initialValue: GenerationViewModel(engine: engine))

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
                .environment(engine)
                .onAppear {
                    appDelegate.onTerminate = {
                        playerViewModel.playerService.shutdown()
                        engine.shutdown()
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
                        NSApplication.AboutPanelOptionKey.applicationVersion: "0.1.0",
                    ])
                }
            }
            CommandGroup(replacing: .newItem) {}
        }
    }
}

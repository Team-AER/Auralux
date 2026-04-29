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

    private let modelContainer: ModelContainer?
    private let modelContainerError: Error?

    init() {
        // Cap the MLX freed-buffer pool. By default MLX retains every buffer it has
        // allocated for reuse, which makes resident memory grow to the high-water mark
        // of the union of every phase (weight load + DiT activations + VAE decode) and
        // never shrink. Halve it under low-memory mode.
        // First launch on a ≤ 16 GiB Mac falls back to the machine class so we
        // don't have to depend on `SettingsViewModel`'s init order to have
        // already persisted its first-launch default.
        let storedLowMemory = UserDefaults.standard.object(forKey: "settings.lowMemoryMode") as? Bool
        let lowMemory = storedLowMemory ?? AppConstants.isLowMemoryMachine
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
            modelContainerError = nil
        } catch {
            modelContainer = nil
            modelContainerError = error
            AppLogger.shared.error("Failed to initialize SwiftData container: \(error)", category: .app)
        }

        AppLogger.shared.info("Auralux launched", category: .app)
    }

    var body: some Scene {
        WindowGroup {
            if let modelContainer {
                ContentView()
                    .frame(minWidth: AppConstants.minimumWindowWidth, minHeight: AppConstants.minimumWindowHeight)
                    .environment(sidebarViewModel)
                    .environment(generationViewModel)
                    .environment(historyViewModel)
                    .environment(playerViewModel)
                    .environment(settingsViewModel)
                    .environment(engine)
                    .modelContainer(modelContainer)
                    .onAppear {
                        appDelegate.onTerminate = {
                            playerViewModel.playerService.shutdown()
                            engine.shutdown()
                        }
                    }
            } else {
                StorageInitErrorView(error: modelContainerError)
                    .frame(minWidth: AppConstants.minimumWindowWidth, minHeight: AppConstants.minimumWindowHeight)
            }
        }
        .defaultSize(width: 1280, height: 840)
        .commands {
            CommandGroup(replacing: .appInfo) {
                Button("About Auralux") {
                    NSApplication.shared.orderFrontStandardAboutPanel(options: [
                        NSApplication.AboutPanelOptionKey.applicationName: AppConstants.appName,
                        NSApplication.AboutPanelOptionKey.applicationVersion: AppConstants.appVersion,
                    ])
                }
            }
            CommandGroup(replacing: .newItem) {}
        }
    }
}

private struct StorageInitErrorView: View {
    let error: Error?

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 48))
                .foregroundStyle(.orange)
            Text("Auralux can't start")
                .font(.title2.bold())
            Text("The on-disk library couldn't be opened. Quit Auralux and try again, or remove the app's container if the problem persists.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
                .frame(maxWidth: 480)
            if let error {
                Text(String(describing: error))
                    .font(.caption.monospaced())
                    .foregroundStyle(.tertiary)
                    .textSelection(.enabled)
                    .padding(.top, 8)
            }
            Button("Quit") { NSApp.terminate(nil) }
                .padding(.top, 8)
        }
        .padding(40)
    }
}

import Foundation

enum AppConstants {
    static let appName = "Auralux"
    static let minimumWindowWidth: Double = 1024
    static let minimumWindowHeight: Double = 768

    /// Marketing version from Info.plist (`CFBundleShortVersionString`),
    /// falling back to "dev" for SPM builds where the plist is absent.
    static var appVersion: String {
        Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "dev"
    }

    // ACE-Step 1.5 model configuration
    static let ditModelName = "acestep-v15-turbo"
    static let llmModelName = "acestep-5Hz-lm-0.6B"
    static let aceStepRepoID = "ACE-Step/Ace-Step1.5"

    // Local storage
    static let modelDirectoryName = "Models"
    static let checkpointDirectoryName = "checkpoints"
    static let generatedDirectoryName = "Generated"
    static let diagnosticsDirectoryName = "Diagnostics"

    static let suggestedTags = [
        "ambient", "cinematic", "lofi", "electronic", "jazz", "piano", "guitar", "synth",
        "uplifting", "melancholic", "driving", "intimate", "orchestral", "vocal", "instrumental",
        "pop", "rock", "hip-hop", "classical", "folk", "r&b", "dance", "country", "reggae",
        "blues", "soul", "metal", "punk", "latin", "world", "acoustic", "experimental"
    ]

    // MARK: - Machine memory tiers
    //
    // ACE-Step 1.5 needs roughly 8–10 GB resident at peak (weight load + DiT
    // activations + VAE decode). Macs at or below 16 GiB need low-memory mode
    // on by default, and machines strictly below 16 GiB get a launch warning
    // because they may swap or run out of memory entirely.

    /// 16 GiB in bytes (matches Apple's binary-GiB reporting).
    static let lowMemoryThresholdBytes: UInt64 = 17_179_869_184

    /// Physical RAM in bytes for the current machine.
    static var physicalMemoryBytes: UInt64 { ProcessInfo.processInfo.physicalMemory }

    /// True when this Mac has ≤ 16 GiB RAM — low-memory mode should be on by default.
    static var isLowMemoryMachine: Bool { physicalMemoryBytes <= lowMemoryThresholdBytes }

    /// True when this Mac has < 16 GiB RAM — warn the user that the app may swap.
    static var isUnderspecMachine: Bool { physicalMemoryBytes < lowMemoryThresholdBytes }
}

import Foundation

enum AppConstants {
    static let appName = "Auralux"
    static let minimumWindowWidth: Double = 1024
    static let minimumWindowHeight: Double = 768

    // Inference server (our thin adapter wrapping ACE-Step 1.5)
    static let inferenceBaseURL = URL(string: "http://127.0.0.1:8765")!

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
}

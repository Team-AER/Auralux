import SwiftUI

struct SettingsView: View {
    @Environment(SettingsViewModel.self) private var viewModel

    var body: some View {
        Form {
            Section("Performance") {
                Picker("Quantization", selection: Bindable(viewModel).quantizationMode) {
                    ForEach(SettingsViewModel.QuantizationMode.allCases) { mode in
                        Text(mode.rawValue.uppercased()).tag(mode)
                    }
                }
                Toggle("Low-memory mode", isOn: Bindable(viewModel).lowMemoryMode)
                Stepper("Max parallel jobs: \(viewModel.maxConcurrentJobs)", value: Bindable(viewModel).maxConcurrentJobs, in: 1...4)
            }

            Section("Inference") {
                Toggle("Auto-start local server", isOn: Bindable(viewModel).autoStartServer)
            }

            Section("Engine") {
                EngineControlPanel()
            }

            Section("Export") {
                Picker("Default format", selection: Bindable(viewModel).defaultExportFormat) {
                    ForEach(AudioExportFormat.allCases.filter(\.isAvailable)) { format in
                        Text(format.rawValue.uppercased()).tag(format)
                    }
                }
            }

            Section {
                ModelSettingsView()
            }
        }
        .padding(20)
    }
}

import SwiftUI

struct SettingsView: View {
    @Environment(SettingsViewModel.self) private var viewModel
    @Environment(NativeInferenceEngine.self) private var engine

    var body: some View {
        ScrollView {
            HStack(alignment: .top, spacing: 24) {
                leftColumn
                rightColumn
            }
            .padding(28)
        }
    }

    // MARK: - Left Column

    private var leftColumn: some View {
        VStack(alignment: .leading, spacing: 16) {
            settingsCard("Performance", systemImage: "gauge.with.dots.needle.50percent") {
                VStack(spacing: 10) {
                    LabeledContent("Quantization") {
                        Text("FP16")
                            .foregroundStyle(.secondary)
                    }
                    Divider()
                    LabeledContent("Low-memory mode") {
                        Toggle("", isOn: Bindable(viewModel).lowMemoryMode)
                            .labelsHidden()
                    }
                    Divider()
                    VStack(alignment: .leading, spacing: 4) {
                        LabeledContent("Load 5Hz LM") {
                            Toggle("", isOn: Bindable(viewModel).useLM)
                                .labelsHidden()
                                .onChange(of: viewModel.useLM) { _, _ in
                                    Task { await engine.loadModels() }
                                }
                        }
                        Text("Loads the 5 Hz audio-token LM (~1.2 GB resident). Reserved for the upcoming text2music-LM mode; off by default.")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }

            settingsCard("Generation", systemImage: "waveform") {
                VStack(spacing: 10) {
                    LabeledContent("DiT variant") {
                        Picker("", selection: Bindable(viewModel).ditVariant) {
                            ForEach(DiTVariant.allCases.filter(\.isAvailable)) { variant in
                                Text(variant.displayName)
                                    .tag(variant)
                            }
                        }
                        .labelsHidden()
                        .frame(width: 220)
                        .onChange(of: viewModel.ditVariant) { _, _ in
                            Task { await engine.loadModels() }
                        }
                    }
                    Divider()
                    LabeledContent("Default mode") {
                        Picker("", selection: Bindable(viewModel).defaultMode) {
                            ForEach(GenerationMode.allCases.filter(\.isImplemented)) { mode in
                                Text(mode.displayName)
                                    .tag(mode)
                            }
                        }
                        .labelsHidden()
                        .frame(width: 220)
                    }
                    Divider()
                    LabeledContent("Default steps") {
                        Stepper(
                            "\(viewModel.defaultNumSteps)",
                            value: Bindable(viewModel).defaultNumSteps,
                            in: 1...viewModel.ditVariant.maxNumSteps
                        )
                        .fixedSize()
                    }
                    Divider()
                    LabeledContent("Default shift") {
                        Picker("", selection: Bindable(viewModel).defaultScheduleShift) {
                            Text("1.0").tag(1.0)
                            Text("2.0").tag(2.0)
                            Text("3.0").tag(3.0)
                        }
                        .pickerStyle(.segmented)
                        .labelsHidden()
                        .frame(width: 160)
                    }
                    Divider()
                    VStack(alignment: .leading, spacing: 4) {
                        LabeledContent("Default CFG scale") {
                            Slider(
                                value: Bindable(viewModel).defaultCfgScale,
                                in: 1.0...20.0
                            )
                            .frame(width: 160)
                            .disabled(!viewModel.ditVariant.respectsCFG)
                            Text(String(format: "%.1f", viewModel.defaultCfgScale))
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(.secondary)
                                .frame(width: 32, alignment: .trailing)
                        }
                        if !viewModel.ditVariant.respectsCFG {
                            Text("Turbo ignores CFG. Switch DiT variant to enable.")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }

            settingsCard("Export", systemImage: "square.and.arrow.up") {
                LabeledContent("Default format") {
                    Picker("", selection: Bindable(viewModel).defaultExportFormat) {
                        ForEach(AudioExportFormat.allCases.filter(\.isAvailable)) { format in
                            Text(format.rawValue.uppercased()).tag(format)
                        }
                    }
                    .labelsHidden()
                    .frame(width: 100)
                }
            }

        }
        .frame(width: 300)
    }

    // MARK: - Right Column

    private var rightColumn: some View {
        VStack(alignment: .leading, spacing: 16) {
            GroupBox {
                ModelSettingsView()
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Card Builder

    @ViewBuilder
    private func settingsCard<Content: View>(
        _ title: String,
        systemImage: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        GroupBox(
            label: Label(title, systemImage: systemImage)
                .font(.headline)
                .padding(.bottom, 4)
        ) {
            content()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

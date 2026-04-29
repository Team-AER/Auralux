import SwiftUI
import UniformTypeIdentifiers

struct ParameterControlsView: View {
    @Environment(GenerationViewModel.self) private var viewModel
    @Environment(SettingsViewModel.self) private var settings

    @State private var importingRefer = false
    @State private var importingSource = false

    var body: some View {
        let bindable = Bindable(viewModel)

        GroupBox(label: parametersHeader) {
            VStack(alignment: .leading, spacing: 12) {
                modePicker(bindable)

                Divider()

                SliderControl(label: "Duration", value: bindable.duration,
                              range: 10...600, unit: "sec",
                              warningThreshold: 240,
                              warningMessage: "May use significant RAM beyond 240 sec.")
                SliderControl(label: "Variance", value: bindable.variance,
                              range: 0...1, unit: "")

                stepsControl(bindable)
                shiftPicker(bindable)
                cfgSlider(bindable)

                if viewModel.mode.requiresReferAudio {
                    audioPicker(
                        label: "Reference audio",
                        url: bindable.referAudioURL,
                        importing: $importingRefer
                    )
                }
                if viewModel.mode.requiresSourceAudio {
                    audioPicker(
                        label: "Source audio",
                        url: bindable.sourceAudioURL,
                        importing: $importingSource
                    )
                }
                if viewModel.mode.requiresRepaintMask {
                    repaintRangeEditor(bindable)
                }

                seedField(bindable)
            }
        }
    }

    private var parametersHeader: some View {
        HStack {
            Text("Parameters")
            Spacer()
            Button {
                viewModel.resetParameters(using: settings)
            } label: {
                Label("Reset", systemImage: "arrow.counterclockwise")
                    .labelStyle(.titleAndIcon)
                    .font(.caption)
            }
            .buttonStyle(.borderless)
            .help("Reset duration, variance, steps, shift, CFG, and seed to defaults for the current model and mode.")
        }
    }

    // MARK: - Mode

    @ViewBuilder
    private func modePicker(_ bindable: Bindable<GenerationViewModel>) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Mode").frame(width: 100, alignment: .leading)
                Picker("", selection: bindable.mode) {
                    ForEach(GenerationMode.allCases.filter(\.isImplemented)) { mode in
                        Text(mode.displayName)
                            .tag(mode)
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
            }
            // mode-specific notes
            switch viewModel.mode {
            case .extract:
                Text("Generates new audio whose timbre matches the reference clip. Reference required.")
                    .font(.caption2).foregroundStyle(.secondary)
            case .cover:
                Text("Re-orchestrates the source audio. Output length follows the source. Reference audio is optional.")
                    .font(.caption2).foregroundStyle(.secondary)
            case .repaint:
                Text("Keeps the source audio and regenerates only the time-ranges below. Output length follows the source.")
                    .font(.caption2).foregroundStyle(.secondary)
            case .text2music, .text2musicLM:
                EmptyView()
            }
        }
    }

    // MARK: - Sampler

    @ViewBuilder
    private func stepsControl(_ bindable: Bindable<GenerationViewModel>) -> some View {
        let maxSteps = Double(settings.ditVariant.maxNumSteps)
        HStack {
            Text("Steps").frame(width: 100, alignment: .leading)
            Slider(value: Binding(
                get: { Double(viewModel.numSteps) },
                set: { bindable.numSteps.wrappedValue = Int($0.rounded()) }
            ), in: 1...maxSteps, step: 1)
            Text("\(viewModel.numSteps)")
                .font(.caption.monospacedDigit())
                .frame(width: 70, alignment: .trailing)
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private func shiftPicker(_ bindable: Bindable<GenerationViewModel>) -> some View {
        HStack {
            Text("Shift").frame(width: 100, alignment: .leading)
            Picker("", selection: bindable.scheduleShift) {
                Text("1.0").tag(1.0)
                Text("2.0").tag(2.0)
                Text("3.0").tag(3.0)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .fixedSize()
            Spacer()
        }
    }

    @ViewBuilder
    private func cfgSlider(_ bindable: Bindable<GenerationViewModel>) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            SliderControl(label: "CFG scale", value: bindable.cfgScale,
                          range: 1.0...20.0, unit: "")
                .disabled(!settings.ditVariant.respectsCFG)
            if !settings.ditVariant.respectsCFG {
                Text("Turbo bakes CFG into the distillation; this slider is a no-op until you switch DiT variants.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - Audio file pickers

    @ViewBuilder
    private func audioPicker(
        label: String,
        url: Binding<URL?>,
        importing: Binding<Bool>
    ) -> some View {
        HStack {
            Text(label).frame(width: 100, alignment: .leading)
            Text(url.wrappedValue?.lastPathComponent ?? "(none)")
                .font(.caption)
                .foregroundStyle(url.wrappedValue == nil ? .secondary : .primary)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer()
            if url.wrappedValue != nil {
                Button("Clear") { url.wrappedValue = nil }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
            }
            Button("Choose…") { importing.wrappedValue = true }
                .buttonStyle(.bordered)
                .controlSize(.small)
        }
        .fileImporter(
            isPresented: importing,
            allowedContentTypes: [.audio, UTType("public.mp3") ?? .audio,
                                  UTType("public.mpeg-4-audio") ?? .audio,
                                  UTType.wav],
            allowsMultipleSelection: false
        ) { result in
            if case let .success(urls) = result, let first = urls.first {
                url.wrappedValue = first
            }
        }
    }

    // MARK: - Repaint range editor
    //
    // Each row is a `[start, end]` second-range to regenerate. Frames outside
    // every range are kept from the source audio; frames inside any range are
    // regenerated. Overlapping ranges are unioned in the engine.
    @ViewBuilder
    private func repaintRangeEditor(_ bindable: Bindable<GenerationViewModel>) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Repaint ranges").frame(width: 100, alignment: .leading)
                Spacer()
                Button {
                    let last = viewModel.repaintRanges.last?.end ?? 0
                    viewModel.repaintRanges.append(
                        RepaintRange(start: last, end: min(last + 5, viewModel.duration))
                    )
                } label: {
                    Label("Add range", systemImage: "plus.circle")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
            if viewModel.repaintRanges.isEmpty {
                Text("No repaint ranges. Add one to regenerate part of the source audio.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(viewModel.repaintRanges.indices, id: \.self) { idx in
                    HStack(spacing: 8) {
                        Text("[\(idx + 1)]").font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                            .frame(width: 24, alignment: .leading)
                        TextField(
                            "start",
                            value: Bindable(viewModel).repaintRanges[idx].start,
                            format: .number.precision(.fractionLength(1))
                        )
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 70)
                        Text("→").foregroundStyle(.secondary)
                        TextField(
                            "end",
                            value: Bindable(viewModel).repaintRanges[idx].end,
                            format: .number.precision(.fractionLength(1))
                        )
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 70)
                        Text("sec").font(.caption2).foregroundStyle(.secondary)
                        Spacer()
                        Button(role: .destructive) {
                            viewModel.repaintRanges.remove(at: idx)
                        } label: {
                            Image(systemName: "trash")
                        }
                        .buttonStyle(.borderless)
                    }
                }
            }
        }
    }

    // MARK: - Seed

    @ViewBuilder
    private func seedField(_ bindable: Bindable<GenerationViewModel>) -> some View {
        HStack {
            Text("Seed").frame(width: 100, alignment: .leading)
            TextField("Random", text: bindable.seedText)
                .textFieldStyle(.roundedBorder)
                .frame(maxWidth: 240)
                .onChange(of: viewModel.seedText) { _, newValue in
                    let digits = newValue.filter(\.isNumber)
                    if digits != newValue { bindable.seedText.wrappedValue = digits }
                }
            Button {
                viewModel.randomizeSeed()
            } label: {
                Image(systemName: "dice")
            }
            .buttonStyle(.borderless)
            .help("Generate a new random seed.")
        }
    }
}

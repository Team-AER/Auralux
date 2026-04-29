import SwiftUI
import UniformTypeIdentifiers

/// Audio-driven generation workspace: cover / repaint / extract.
///
/// Shares state with the main Generate tab via `GenerationViewModel`, so
/// switching sidebar sections preserves the current request. The mode picker
/// here is constrained to audio-in modes; text-only generation lives on the
/// Generate tab.
struct AudioImportView: View {
    @Environment(GenerationViewModel.self) private var viewModel
    @Environment(SettingsViewModel.self) private var settings
    @Environment(NativeInferenceEngine.self) private var engine
    @Environment(\.modelContext) private var modelContext

    @State private var tagText = ""
    @State private var importingSource = false
    @State private var importingRefer = false
    @State private var completionDismissTask: Task<Void, Never>?

    private static let audioInModes: [GenerationMode] = [.extract, .cover, .repaint]

    private var generateDisabled: Bool {
        viewModel.state.isBusy
            || !engine.modelState.isReady
            || engine.isGenerating
            || (viewModel.mode.requiresSourceAudio && viewModel.sourceAudioURL == nil)
            || (viewModel.mode.requiresReferAudio && viewModel.referAudioURL == nil)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header

                modePicker
                audioInputs
                if viewModel.mode.requiresRepaintMask {
                    repaintRangeEditor
                }

                GroupBox("Prompt") {
                    TextEditor(text: Bindable(viewModel).prompt)
                        .font(.body)
                        .frame(minHeight: 80)
                }

                TagEditorView(tags: viewModel.tags, draftTag: $tagText) { tag in
                    viewModel.addTag(tag)
                } onRemoveTag: { tag in
                    viewModel.removeTag(tag)
                }

                LyricEditorView(lyrics: Bindable(viewModel).lyrics)

                samplerControls

                generateRow
            }
            .padding(20)
        }
        .onAppear {
            // Snap into an audio-in mode the first time we land here so the
            // tab actually does what its name says.
            if !Self.audioInModes.contains(viewModel.mode) {
                viewModel.mode = .extract
            }
        }
    }

    // MARK: - Header

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Audio to Audio")
                .font(.title2.weight(.semibold))
            Text("Generate from a source or reference clip — cover, repaint, or extract timbre.")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - Mode

    private var modePicker: some View {
        GroupBox("Mode") {
            VStack(alignment: .leading, spacing: 6) {
                Picker("", selection: Bindable(viewModel).mode) {
                    ForEach(Self.audioInModes) { mode in
                        Text(mode.displayName).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()

                Text(modeDescription)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var modeDescription: String {
        switch viewModel.mode {
        case .extract:
            return "Generates new audio whose timbre matches the reference clip. Reference required."
        case .cover:
            return "Re-orchestrates the source audio. Output length follows the source. Reference is optional."
        case .repaint:
            return "Keeps the source audio and regenerates only the time-ranges below."
        case .text2music, .text2musicLM:
            return ""
        }
    }

    // MARK: - Audio inputs

    private var audioInputs: some View {
        GroupBox("Audio inputs") {
            VStack(alignment: .leading, spacing: 14) {
                if viewModel.mode.requiresSourceAudio {
                    audioSlot(
                        title: "Source audio",
                        url: Bindable(viewModel).sourceAudioURL,
                        importing: $importingSource,
                        helper: "The clip to cover or repaint. Output length follows this file."
                    )
                }
                if viewModel.mode.requiresReferAudio {
                    audioSlot(
                        title: viewModel.mode == .extract ? "Reference audio" : "Reference audio (optional)",
                        url: Bindable(viewModel).referAudioURL,
                        importing: $importingRefer,
                        helper: viewModel.mode == .extract
                            ? "The timbre target. Required."
                            : "Biases the cover toward this clip's timbre."
                    )
                }
            }
        }
    }

    @ViewBuilder
    private func audioSlot(
        title: String,
        url: Binding<URL?>,
        importing: Binding<Bool>,
        helper: String
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(title).font(.callout.weight(.medium))
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
            if let resolved = url.wrappedValue {
                Label(resolved.lastPathComponent, systemImage: "waveform")
                    .font(.caption.monospaced())
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(.tint.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
            } else {
                AudioDropZone(selectedFileURL: url)
            }
            Text(helper).font(.caption2).foregroundStyle(.secondary)
        }
        .fileImporter(
            isPresented: importing,
            allowedContentTypes: [.audio, UTType.wav, UTType("public.mp3") ?? .audio,
                                  UTType("public.mpeg-4-audio") ?? .audio],
            allowsMultipleSelection: false
        ) { result in
            if case let .success(urls) = result, let first = urls.first {
                url.wrappedValue = first
            }
        }
    }

    // MARK: - Repaint ranges

    private var repaintRangeEditor: some View {
        GroupBox("Repaint ranges") {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Frames inside any range are regenerated. Everything else is kept from the source.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
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
                    Text("No ranges yet. Add one to repaint a slice of the source audio.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(viewModel.repaintRanges.indices, id: \.self) { idx in
                        HStack(spacing: 8) {
                            Text("[\(idx + 1)]")
                                .font(.caption.monospacedDigit())
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
    }

    // MARK: - Sampler

    private var samplerControls: some View {
        GroupBox(label: HStack {
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
        }) {
            let bindable = Bindable(viewModel)
            VStack(alignment: .leading, spacing: 12) {
                if !viewModel.mode.requiresSourceAudio {
                    SliderControl(label: "Duration", value: bindable.duration,
                                  range: 10...600, unit: "sec",
                                  warningThreshold: 240,
                                  warningMessage: "May use significant RAM beyond 240 sec.")
                }
                SliderControl(label: "Variance", value: bindable.variance,
                              range: 0...1, unit: "")

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

                HStack {
                    Text("Seed").frame(width: 100, alignment: .leading)
                    TextField("Random", text: bindable.seedText)
                        .textFieldStyle(.roundedBorder)
                        .frame(maxWidth: 240)
                        .onChange(of: viewModel.seedText) { _, newValue in
                            let digits = newValue.filter(\.isNumber)
                            if digits != newValue { bindable.seedText.wrappedValue = digits }
                        }
                }
            }
        }
    }

    // MARK: - Generate row

    private var generateRow: some View {
        HStack {
            Button("Generate") {
                viewModel.generate(in: modelContext)
            }
            .keyboardShortcut("g", modifiers: [.command])
            .disabled(generateDisabled)

            Button("Cancel") {
                viewModel.cancel()
            }
            .disabled(!viewModel.state.isBusy)

            Spacer()

            switch viewModel.state {
            case .preparing, .generating:
                VStack(alignment: .trailing, spacing: 4) {
                    ProgressView(value: viewModel.progress)
                        .frame(width: 200)
                    if !viewModel.progressMessage.isEmpty {
                        Text(viewModel.progressMessage)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }
            case .completed:
                Label("Done", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                    .onAppear {
                        completionDismissTask?.cancel()
                        completionDismissTask = Task {
                            try? await Task.sleep(for: .seconds(3))
                            if !Task.isCancelled { viewModel.state = .idle }
                        }
                    }
                    .onDisappear { completionDismissTask?.cancel() }
            case .failed(let message):
                HStack(spacing: 6) {
                    Label(message, systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                    Button {
                        viewModel.state = .idle
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }
            case .idle:
                if !engine.modelState.isReady {
                    Label("Engine not ready", systemImage: "moon.zzz")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }
}

import SwiftUI

struct PlayerView: View {
    let track: GeneratedTrack

    @Environment(PlayerViewModel.self) private var viewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 6) {
                Text(track.title)
                    .font(.title2.weight(.semibold))
                    .accessibilityIdentifier("track-title")
                Text(track.createdAt.formatted(date: .abbreviated, time: .shortened))
                    .foregroundStyle(.secondary)
            }

            if let error = viewModel.errorMessage {
                Label(error, systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.red)
                    .font(.callout)
            }

            WaveformView(
                progress: viewModel.progress,
                samples: viewModel.waveformSamples,
                onSeek: { fraction in viewModel.seek(to: fraction) }
            )
            .frame(height: 120)

            SpectrumAnalyzerView(
                magnitudes: viewModel.playerService.spectrumMagnitudes,
                isPlaying: viewModel.isPlaying
            )
            .frame(height: 90)

            HStack(spacing: 12) {
                Button(viewModel.isPlaying ? "Pause" : "Play") {
                    viewModel.playPause()
                }
                .keyboardShortcut(.space, modifiers: [])
                .accessibilityLabel(viewModel.isPlaying ? "Pause" : "Play")
                .accessibilityIdentifier("play-pause-button")
                .disabled(viewModel.errorMessage != nil)

                Button("Stop") {
                    viewModel.stop()
                }
                .disabled(viewModel.errorMessage != nil)

                Button("Capture RCA Log") {
                    _ = viewModel.capturePlaybackDiagnostics(reason: "manual_capture_button")
                }
                .help("Write a playback diagnostics snapshot to Application Support/Auralux/Diagnostics")

                @Bindable var vm = viewModel
                Toggle("Loop", isOn: $vm.isLooping)
                    .toggleStyle(.switch)
                    .frame(width: 110)

                Spacer()

                Text(timecode)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }

            Spacer(minLength: 0)
        }
        .padding(20)
        .task(id: track.id) {
            // Defer audio engine init to the next run-loop turn so it
            // doesn't collide with the SwiftUI/SwiftData update cycle
            // that surfaces this view (causes _dispatch_assert_queue_fail).
            try? await Task.sleep(for: .milliseconds(200))
            guard !Task.isCancelled else { return }
            viewModel.load(path: track.audioFilePath)
        }
    }

    private var timecode: String {
        let current = Int(viewModel.currentTime)
        let total = Int(viewModel.duration)
        return "\(format(seconds: current)) / \(format(seconds: total))"
    }

    private func format(seconds: Int) -> String {
        String(format: "%02d:%02d", max(0, seconds / 60), max(0, seconds % 60))
    }
}

import SwiftUI

struct PlayerView: View {
    let track: GeneratedTrack

    @Environment(PlayerViewModel.self) private var viewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 6) {
                Text(track.title)
                    .font(.title2.weight(.semibold))
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

                Button("Stop") {
                    viewModel.stop()
                }

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
        .onAppear {
            viewModel.load(path: track.audioFilePath)
        }
        .onChange(of: track.id) { _, _ in
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

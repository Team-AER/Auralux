import SwiftUI

struct LyricEditorView: View {
    @Binding var lyrics: String
    @Environment(SettingsViewModel.self) private var settings

    var body: some View {
        GroupBox("Lyrics") {
            TextEditor(text: $lyrics)
                .font(.system(.body, design: .monospaced))
                .frame(minHeight: 180)
                .overlay(alignment: .topLeading) {
                    if lyrics.isEmpty {
                        Text("[verse]\nWrite your lyrics here...")
                            .foregroundStyle(.secondary)
                            .padding(8)
                            .allowsHitTesting(false)
                    }
                }

            if settings.ditVariant.usesCFGDistillation && !lyrics.isEmpty {
                Text("Turbo models may skip later verses — switch to SFT for closer lyric following.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.top, 2)
            }
        }
    }
}

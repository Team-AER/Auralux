import SwiftUI

struct TagEditorView: View {
    let tags: [String]
    @Binding var draftTag: String
    var onAddTag: (String) -> Void
    var onRemoveTag: (String) -> Void

    var body: some View {
        GroupBox("Tags") {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    TextField("Add genre, instrument, or mood", text: $draftTag)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit(addDraftTag)

                    Button("Add") {
                        addDraftTag()
                    }
                    .disabled(draftTag.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }

                FlowTagList(tags: tags, onRemoveTag: onRemoveTag)

                ScrollView(.horizontal) {
                    HStack {
                        ForEach(AppConstants.suggestedTags, id: \.self) { tag in
                            Button(tag) {
                                onAddTag(tag)
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                        }
                    }
                }
            }
        }
    }

    private func addDraftTag() {
        let value = draftTag.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty else { return }
        onAddTag(value)
        draftTag = ""
    }
}

private struct FlowTagList: View {
    let tags: [String]
    var onRemoveTag: (String) -> Void

    var body: some View {
        if tags.isEmpty {
            Text("No tags yet")
                .foregroundStyle(.secondary)
        } else {
            HStack {
                ForEach(tags, id: \.self) { tag in
                    TagChip(tag: tag) {
                        onRemoveTag(tag)
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

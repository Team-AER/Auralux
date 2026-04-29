import SwiftUI

struct TagChip: View {
    let tag: String
    var onDelete: (() -> Void)?

    var body: some View {
        HStack(spacing: 6) {
            Text(tag)
                .font(.caption.weight(.medium))
            if let onDelete {
                Button {
                    onDelete()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Remove \(tag)")
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(.thinMaterial, in: Capsule())
    }
}

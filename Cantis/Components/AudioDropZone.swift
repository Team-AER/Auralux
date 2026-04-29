import UniformTypeIdentifiers
import SwiftUI

struct AudioDropZone: View {
    @Binding var selectedFileURL: URL?

    var body: some View {
        RoundedRectangle(cornerRadius: 14)
            .strokeBorder(style: StrokeStyle(lineWidth: 1.5, dash: [6]))
            .fill(.clear)
            .overlay {
                VStack(spacing: 8) {
                    Image(systemName: "square.and.arrow.down.on.square")
                        .font(.title2)
                    Text("Drop audio file here")
                }
                .foregroundStyle(.secondary)
            }
            .frame(height: 140)
            .onDrop(of: [.audio], isTargeted: nil) { providers in
                guard let provider = providers.first else { return false }
                provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, _ in
                    guard let data = item as? Data,
                          let url = URL(dataRepresentation: data, relativeTo: nil) else { return }
                    DispatchQueue.main.async {
                        selectedFileURL = url
                    }
                }
                return true
            }
    }
}

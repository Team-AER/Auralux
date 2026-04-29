import SwiftData
import SwiftUI

struct HistoryBrowserView: View {
    @Environment(HistoryViewModel.self) private var viewModel
    @Environment(\.modelContext) private var modelContext
    @State private var pendingDelete: GeneratedTrack?
    @State private var showDeleteAllConfirm = false

    var body: some View {
        VStack(spacing: 12) {
            HStack {
                TextField("Search history", text: Bindable(viewModel).query)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit {
                        viewModel.refresh(context: modelContext)
                    }

                if !viewModel.query.isEmpty {
                    Button("Clear") {
                        viewModel.query = ""
                        viewModel.refresh(context: modelContext)
                    }
                    .controlSize(.small)
                }

                Button("Refresh") {
                    viewModel.refresh(context: modelContext)
                }

                Button(role: .destructive) {
                    showDeleteAllConfirm = true
                } label: {
                    Label("Delete All", systemImage: "trash")
                }
                .disabled(viewModel.tracks.isEmpty)
            }

            if viewModel.tracks.isEmpty {
                if viewModel.query.isEmpty {
                    ContentUnavailableView(
                        "No Generations Yet",
                        systemImage: "music.note.list",
                        description: Text("Generated tracks will appear here.")
                    )
                } else {
                    ContentUnavailableView(
                        "No Matches",
                        systemImage: "magnifyingglass",
                        description: Text("No tracks match \"\(viewModel.query)\".")
                    )
                }
            } else {
                List(viewModel.tracks, selection: Bindable(viewModel).selectedTrack) { track in
                    HistoryItemView(track: track)
                        .tag(track)
                        .contextMenu {
                            Button(track.isFavorite ? "Remove Favorite" : "Favorite") {
                                viewModel.toggleFavorite(track, context: modelContext)
                            }
                            Divider()
                            Button("Delete", role: .destructive) {
                                pendingDelete = track
                            }
                        }
                        .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                            Button(role: .destructive) {
                                pendingDelete = track
                            } label: {
                                Label("Delete", systemImage: "trash")
                            }
                        }
                }
                .listStyle(.inset)
            }
        }
        .padding(20)
        .onAppear {
            viewModel.refresh(context: modelContext)
        }
        .onChange(of: viewModel.query) { _, _ in
            viewModel.refresh(context: modelContext)
        }
        .confirmationDialog(
            pendingDelete.map { "Delete \($0.title)?" } ?? "",
            isPresented: Binding(
                get: { pendingDelete != nil },
                set: { if !$0 { pendingDelete = nil } }
            ),
            presenting: pendingDelete
        ) { track in
            Button("Delete", role: .destructive) {
                viewModel.delete(track, context: modelContext)
            }
            Button("Cancel", role: .cancel) { }
        } message: { _ in
            Text("This removes the track and its audio file. This cannot be undone.")
        }
        .confirmationDialog(
            "Delete all history?",
            isPresented: $showDeleteAllConfirm
        ) {
            Button("Delete All", role: .destructive) {
                viewModel.deleteAll(context: modelContext)
            }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("Removes every generated track and its audio file. This cannot be undone.")
        }
    }
}

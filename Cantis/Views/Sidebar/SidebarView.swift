import SwiftData
import SwiftUI

struct SidebarView: View {
    @Environment(SidebarViewModel.self) private var viewModel

    var body: some View {
        @Bindable var viewModel = viewModel
        List(selection: $viewModel.selectedSection) {
            Section("Workspace") {
                ForEach(SidebarSection.allCases) { section in
                    NavigationLink(value: section) {
                        Label(section.title, systemImage: icon(for: section))
                    }
                }
            }

            Section("Presets") {
                PresetListView()
            }

            Section("Recent") {
                RecentListView()
            }
        }
        .listStyle(.sidebar)
    }

    private func icon(for section: SidebarSection) -> String {
        switch section {
        case .generate:
            "wand.and.stars"
        case .history:
            "clock.arrow.circlepath"
        case .audioToAudio:
            "waveform.path.badge.plus"
        case .settings:
            "gearshape"
        }
    }
}

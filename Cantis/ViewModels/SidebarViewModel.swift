import Foundation
import Observation

enum SidebarSection: String, CaseIterable, Identifiable {
    case generate
    case history
    case audioToAudio
    case settings

    var id: String { rawValue }

    var title: String {
        switch self {
        case .generate: "Generate"
        case .history: "History"
        case .audioToAudio: "Audio to Audio"
        case .settings: "Settings"
        }
    }
}

@MainActor
@Observable
final class SidebarViewModel {
    var selectedSection: SidebarSection? = .generate
}

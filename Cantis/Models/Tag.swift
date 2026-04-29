import Foundation
import SwiftData

@Model
final class Tag {
    @Attribute(.unique) var id: UUID
    @Attribute(.unique) var name: String
    var category: String
    var createdAt: Date

    init(id: UUID = UUID(), name: String, category: String = "custom", createdAt: Date = .now) {
        self.id = id
        self.name = name
        self.category = category
        self.createdAt = createdAt
    }
}

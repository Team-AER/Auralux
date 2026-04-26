// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "Auralux",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .executable(name: "Auralux", targets: ["Auralux"])
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-collections.git", from: "1.1.0"),
        .package(url: "https://github.com/markiv/SwiftUI-Shimmer.git", from: "1.5.1")
    ],
    targets: [
        .executableTarget(
            name: "Auralux",
            dependencies: [
                .product(name: "Collections", package: "swift-collections"),
                .product(name: "Shimmer", package: "SwiftUI-Shimmer")
            ],
            path: "Auralux",
            exclude: [
                "Info.plist",
                "Entitlements.plist"
            ],
            resources: [
                .process("Resources")
            ]
        ),
        .testTarget(
            name: "AuraluxTests",
            dependencies: ["Auralux"],
            path: "AuraluxTests"
        )
    ]
)

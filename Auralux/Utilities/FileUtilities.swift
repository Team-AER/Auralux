import Foundation

enum FileUtilities {
    static var appSupportDirectory: URL {
        guard let baseURL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else {
            fatalError("Application Support directory is unavailable — cannot proceed.")
        }
        let appURL = baseURL.appendingPathComponent(AppConstants.appName, isDirectory: true)
        if !FileManager.default.fileExists(atPath: appURL.path) {
            try? FileManager.default.createDirectory(at: appURL, withIntermediateDirectories: true)
        }
        return appURL
    }

    static var modelDirectory: URL {
        let url = appSupportDirectory.appendingPathComponent(AppConstants.modelDirectoryName, isDirectory: true)
        if !FileManager.default.fileExists(atPath: url.path) {
            try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        }
        return url
    }

    static var generatedAudioDirectory: URL {
        let url = appSupportDirectory.appendingPathComponent(AppConstants.generatedDirectoryName, isDirectory: true)
        if !FileManager.default.fileExists(atPath: url.path) {
            try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        }
        return url
    }

    static var diagnosticsDirectory: URL {
        let url = appSupportDirectory.appendingPathComponent(AppConstants.diagnosticsDirectoryName, isDirectory: true)
        if !FileManager.default.fileExists(atPath: url.path) {
            try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        }
        return url
    }

    static func copyToGeneratedDirectory(from sourceURL: URL, name: String) throws -> URL {
        let destination = generatedAudioDirectory.appendingPathComponent(name)
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.copyItem(at: sourceURL, to: destination)
        return destination
    }

    // MARK: - Path Resolution

    /// Converts an absolute audio path to a path relative to the generated audio
    /// directory. Stores just the filename so paths survive sandbox container changes.
    static func relativeAudioPath(from absolutePath: String) -> String {
        let url = URL(fileURLWithPath: absolutePath)
        let genDir = generatedAudioDirectory.path
        if absolutePath.hasPrefix(genDir) {
            let relative = String(absolutePath.dropFirst(genDir.count))
            return relative.hasPrefix("/") ? String(relative.dropFirst()) : relative
        }
        return url.lastPathComponent
    }

    /// Resolves a stored audio path (which may be absolute for backwards
    /// compatibility or relative) to a full URL on the current system.
    static func resolveAudioPath(_ storedPath: String) -> URL {
        if storedPath.hasPrefix("/") {
            let url = URL(fileURLWithPath: storedPath)
            if FileManager.default.fileExists(atPath: url.path) {
                return url
            }
            let fallback = generatedAudioDirectory.appendingPathComponent(url.lastPathComponent)
            return fallback
        }
        return generatedAudioDirectory.appendingPathComponent(storedPath)
    }
}

import SwiftUI

/// A standalone window that displays real-time application logs.
/// Supports filtering by level and category, auto-scrolling,
/// and copy / export actions.
struct LogViewerView: View {
    private let logger = AppLogger.shared

    @State private var filterLevel: LogLevel? = nil
    @State private var filterCategory: LogCategory? = nil
    @State private var searchText = ""
    @State private var autoScroll = true
    @State private var scrollTask: Task<Void, Never>?

    private var filteredEntries: [LogEntry] {
        logger.entries.filter { entry in
            if let lvl = filterLevel, entry.level != lvl { return false }
            if let cat = filterCategory, entry.category != cat { return false }
            if !searchText.isEmpty,
               !entry.message.localizedCaseInsensitiveContains(searchText) {
                return false
            }
            return true
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            toolbar
            Divider()
            logList
            Divider()
            statusBar
        }
        .frame(minWidth: 600, minHeight: 300)
    }

    // MARK: - Toolbar

    private var toolbar: some View {
        HStack(spacing: 10) {
            // Level filter
            Picker("Level", selection: $filterLevel) {
                Text("All Levels").tag(LogLevel?.none)
                ForEach(LogLevel.allCases, id: \.self) { level in
                    Text(level.rawValue).tag(LogLevel?.some(level))
                }
            }
            .frame(width: 130)

            // Category filter
            Picker("Category", selection: $filterCategory) {
                Text("All Categories").tag(LogCategory?.none)
                ForEach(LogCategory.allCases, id: \.self) { cat in
                    Text(cat.rawValue).tag(LogCategory?.some(cat))
                }
            }
            .frame(width: 150)

            // Search
            TextField("Search logs...", text: $searchText)
                .textFieldStyle(.roundedBorder)
                .frame(maxWidth: 220)

            Spacer()

            Toggle("Auto-scroll", isOn: $autoScroll)
                .toggleStyle(.checkbox)

            Button {
                let text = logger.exportText()
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(text, forType: .string)
            } label: {
                Image(systemName: "doc.on.doc")
            }
            .help("Copy all logs to clipboard")

            Button {
                logger.clear()
            } label: {
                Image(systemName: "trash")
            }
            .help("Clear logs")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    // MARK: - Log List

    private var logList: some View {
        ScrollViewReader { proxy in
            List(filteredEntries) { entry in
                logRow(entry)
                    .id(entry.id)
                    .listRowSeparator(.hidden)
                    .listRowInsets(EdgeInsets(top: 1, leading: 8, bottom: 1, trailing: 8))
            }
            .listStyle(.plain)
            .font(.system(.caption, design: .monospaced))
            .onChange(of: filteredEntries.count) { _, _ in
                guard autoScroll else { return }
                scrollTask?.cancel()
                scrollTask = Task {
                    try? await Task.sleep(for: .milliseconds(250))
                    guard !Task.isCancelled, let last = filteredEntries.last else { return }
                    proxy.scrollTo(last.id, anchor: .bottom)
                }
            }
        }
    }

    private func logRow(_ entry: LogEntry) -> some View {
        HStack(alignment: .top, spacing: 6) {
            Text(entry.formattedTimestamp)
                .foregroundStyle(.secondary)
                .frame(width: 90, alignment: .leading)

            Text(entry.level.rawValue)
                .foregroundStyle(colorForLevel(entry.level))
                .fontWeight(.medium)
                .frame(width: 44, alignment: .leading)

            Text(entry.category.rawValue)
                .foregroundStyle(.secondary)
                .frame(width: 80, alignment: .leading)

            Text(entry.message)
                .foregroundStyle(.primary)
                .lineLimit(nil)
                .textSelection(.enabled)
        }
    }

    private func colorForLevel(_ level: LogLevel) -> Color {
        switch level {
        case .debug:   .secondary
        case .info:    .blue
        case .warning: .orange
        case .error:   .red
        }
    }

    // MARK: - Status Bar

    private var statusBar: some View {
        HStack {
            Text("\(filteredEntries.count) / \(logger.entries.count) entries")
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 4)
    }
}

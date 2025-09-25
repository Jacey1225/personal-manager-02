import SwiftUI

// MARK: - Shared Event Models

struct SelectableEvent: Identifiable {
    let id = UUID()
    let eventName: String
    let eventId: String
    let startDate: String?
    let endDate: String?
}

struct ResponseRequest {
    var userId: String?
    var status: String
    var message: String
    var eventRequested: [String: Any]
    var calendarInsights: [String: Any]?
    
    init(userId: String? = nil, status: String, message: String, eventRequested: [String: Any], calendarInsights: [String: Any]? = nil) {
        self.userId = userId
        self.status = status
        self.message = message
        self.eventRequested = eventRequested
        self.calendarInsights = calendarInsights
    }
}

// MARK: - Shared Event Selection View

struct EventSelectionView: View {
    let selectableEvents: [SelectableEvent]
    let onEventSelected: (SelectableEvent) -> Void
    let onCancel: () -> Void
    
    var body: some View {
        NavigationView {
            List(selectableEvents) { event in
                VStack(alignment: .leading, spacing: 4) {
                    Text(event.eventName)
                        .font(.headline)
                    if let startDate = event.startDate {
                        Text("Start: \(formatDate(startDate))")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    if let endDate = event.endDate {
                        Text("End: \(formatDate(endDate))")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                .contentShape(Rectangle())
                .onTapGesture {
                    onEventSelected(event)
                }
            }
            .navigationTitle("Select Event")
            .navigationBarTitleDisplayMode(.inline)
            .navigationBarItems(
                trailing: Button("Cancel") {
                    onCancel()
                }
            )
        }
    }
    
    private func formatDate(_ dateString: String) -> String {
        let formatter = ISO8601DateFormatter()
        if let date = formatter.date(from: dateString) {
            let displayFormatter = DateFormatter()
            displayFormatter.dateStyle = .short
            displayFormatter.timeStyle = .short
            return displayFormatter.string(from: date)
        }
        return dateString
    }
}

import SwiftUI

struct TaskEvent: Codable, Identifiable {
    let id = UUID()
    let event_name: String
    let start_time: String
    let end_time: String
    let event_id: String
    
    // Custom coding keys to match the API response
    private enum CodingKeys: String, CodingKey {
        case event_name
        case start_time
        case end_time
        case event_id
    }
    
    var formattedStartTime: String {
        return formatDateTime(start_time)
    }
    
    var formattedEndTime: String {
        return formatDateTime(end_time)
    }
    
    private func formatDateTime(_ dateString: String) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        
        if let date = formatter.date(from: dateString) {
            let displayFormatter = DateFormatter()
            displayFormatter.dateFormat = "MMM d, yyyy h:mm a"
            return displayFormatter.string(from: date)
        }
        return dateString
    }
}

struct TasksListView: View {
    let userId: String
    @State private var events: [TaskEvent] = []
    @State private var isLoading = false
    @State private var errorMessage = ""
    @State private var showingError = false
    @State private var deletingEventId: String? = nil
    
    var body: some View {
        NavigationView {
            VStack {
                if isLoading {
                    VStack(spacing: 20) {
                        ProgressView("Loading events...")
                        Text("Fetching your tasks and events")
                            .foregroundColor(.secondary)
                            .font(.caption)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if events.isEmpty {
                    VStack(spacing: 20) {
                        Image(systemName: "calendar.badge.exclamationmark")
                            .font(.system(size: 60))
                            .foregroundColor(.gray)
                        
                        Text("No Events Found")
                            .font(.title2)
                            .fontWeight(.semibold)
                        
                        Text("You don't have any events or tasks scheduled.")
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                        
                        Button("Refresh") {
                            fetchEvents()
                        }
                        .buttonStyle(.bordered)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .padding()
                } else {
                    List {
                        ForEach(events) { event in
                            EventRowView(
                                event: event,
                                isDeleting: deletingEventId == event.event_id,
                                onDelete: { deleteEvent(event) }
                            )
                        }
                    }
                    .refreshable {
                        fetchEvents()
                    }
                }
            }
            .navigationTitle("Tasks & Events")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Refresh") {
                        fetchEvents()
                    }
                    .disabled(isLoading)
                }
            }
            .onAppear {
                fetchEvents()
            }
            .alert("Error", isPresented: $showingError) {
                Button("OK") {
                    errorMessage = ""
                }
                Button("Retry") {
                    fetchEvents()
                }
            } message: {
                Text(errorMessage)
            }
        }
    }
    
    private func fetchEvents() {
        isLoading = true
        errorMessage = ""
        
        guard let url = URL(string: "https://f8cb321bfd70.ngrok-free.app/scheduler/list_events") else {
            errorMessage = "Invalid URL"
            isLoading = false
            showingError = true
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let requestBody: [String: Any] = [
            "event_details": [
                "event_name": "",
                "datetime_obj": [
                    "target_datetimes": []
                ],
                "action": "list",
                "response": ""
            ] as [String: Any],
            "user_id": userId
        ]
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            errorMessage = "Failed to encode request"
            isLoading = false
            showingError = true
            return
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                isLoading = false
                
                if let error = error {
                    errorMessage = "Network error: \(error.localizedDescription)"
                    showingError = true
                    return
                }
                
                guard let data = data else {
                    errorMessage = "No data received"
                    showingError = true
                    return
                }
                
                if let jsonString = String(data: data, encoding: .utf8) {
                    print("Raw API Response: \(jsonString)")
                }
                
                do {
                    if let jsonArray = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
                        print("Parsed JSON Array: \(jsonArray)")
                        
                        var parsedEvents: [TaskEvent] = []
                        
                        for eventDict in jsonArray {
                            if let eventName = eventDict["event_name"] as? String,
                               let startTime = eventDict["start_time"] as? String,
                               let endTime = eventDict["end_time"] as? String,
                               let eventId = eventDict["event_id"] as? String {
                                
                                let taskEvent = TaskEvent(
                                    event_name: eventName,
                                    start_time: startTime,
                                    end_time: endTime,
                                    event_id: eventId
                                )
                                parsedEvents.append(taskEvent)
                            }
                        }
                        
                        events = parsedEvents
                        print("Successfully parsed \(events.count) events")
                        
                    } else {
                        errorMessage = "Response is not in expected array format"
                        showingError = true
                    }
                    
                } catch {
                    errorMessage = "Failed to parse events: \(error.localizedDescription)"
                    showingError = true
                    print("JSON parsing error: \(error)")
                }
            }
        }.resume()
    }
    
    private func deleteEvent(_ event: TaskEvent) {
        deletingEventId = event.event_id
        
        guard let url = URL(string: "https://f8cb321bfd70.ngrok-free.app/scheduler/delete_event/\(event.event_id)") else {
            errorMessage = "Invalid URL"
            showingError = true
            deletingEventId = nil
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        // Create complete responseRequest structure with proper event data mapping
        let responseRequest: [String: Any] = [
            "user_id": userId,
            "status": "request event ID",
            "message": "Event '\(event.event_name)' selected for deletion from task list",
            "event_requested": [
                "event_name": event.event_name,
                "target_dates": createTargetDatetimes(from: event),
                "action": "delete",
                "response": "Event '\(event.event_name)' deleted successfully"
            ],
            "calendar_insights": [
                "matching_events": [
                    [
                        "event_name": event.event_name,
                        "event_id": event.event_id,
                        "start": event.start_time,
                        "end": event.end_time.isEmpty ? event.start_time : event.end_time,
                        "is_event": determineIsEvent(from: event)
                    ]
                ],
                "is_event": determineIsEvent(from: event),
                "template": [:],
                "selected_event_id": event.event_id
            ]
        ]
        
        print("Sending delete request with event data:")
        print("- Event Name: \(event.event_name)")
        print("- Event ID: \(event.event_id)")
        print("- Start Time: \(event.start_time)")
        print("- End Time: \(event.end_time)")
        print("- Is Event: \(determineIsEvent(from: event))")
        print("- Full Request: \(responseRequest)")
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: responseRequest)
        } catch {
            errorMessage = "Failed to encode request"
            showingError = true
            deletingEventId = nil
            return
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                deletingEventId = nil
                
                if let error = error {
                    errorMessage = "Network error: \(error.localizedDescription)"
                    showingError = true
                    return
                }
                
                guard let data = data else {
                    errorMessage = "No data received"
                    showingError = true
                    return
                }
                
                // Log the raw response for debugging
                if let responseString = String(data: data, encoding: .utf8) {
                    print("Delete API Response: \(responseString)")
                }
                
                do {
                    if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                       let status = json["status"] as? String {
                        if status == "success" {
                            // Remove the deleted event from the list
                            events.removeAll { $0.event_id == event.event_id }
                            print("Successfully deleted event: \(event.event_name)")
                        } else {
                            errorMessage = json["message"] as? String ?? "Failed to delete event"
                            showingError = true
                            print("Delete failed: \(errorMessage)")
                        }
                    } else {
                        errorMessage = "Invalid response format"
                        showingError = true
                        print("Invalid response format from delete API")
                    }
                } catch {
                    errorMessage = "Failed to parse response: \(error.localizedDescription)"
                    showingError = true
                    print("Failed to parse delete response: \(error)")
                }
            }
        }.resume()
    }
    
    // MARK: - Helper Functions for Event Details Compilation (Updated)
    
    private func createTargetDatetimes(from event: TaskEvent) -> [[String]] {
        // Convert the event's start and end times to the format expected by the API
        let startTime = event.start_time
        let endTime = event.end_time
        
        print("Creating target datetimes for event: \(event.event_name)")
        print("Start time: \(startTime)")
        print("End time: \(endTime)")
        
        // Return as array of arrays containing start and end time strings
        if endTime.isEmpty || endTime == startTime {
            // This is likely a task (no end time)
            print("Treating as task (no end time)")
            return [[startTime]]
        } else {
            // This is an event with start and end times
            print("Treating as event (has end time)")
            return [[startTime, endTime]]
        }
    }
    
    private func determineIsEvent(from event: TaskEvent) -> Bool {
        // Determine if this is an event (has end time) or task (no end time)
        let isEvent = !event.end_time.isEmpty && event.end_time != event.start_time
        print("Event '\(event.event_name)' is_event: \(isEvent)")
        return isEvent
    }
}

struct EventRowView: View {
    let event: TaskEvent
    let isDeleting: Bool
    let onDelete: () -> Void
    
    var body: some View {
        HStack(spacing: 12) {
            // Delete button
            Button(action: onDelete) {
                if isDeleting {
                    ProgressView()
                        .scaleEffect(0.8)
                        .frame(width: 20, height: 20)
                } else {
                    Image(systemName: "trash")
                        .foregroundColor(.red)
                        .font(.system(size: 16, weight: .medium))
                }
            }
            .frame(width: 30, height: 30)
            .disabled(isDeleting)
            
            // Event details
            VStack(alignment: .leading, spacing: 4) {
                Text(event.event_name)
                    .font(.headline)
                    .fontWeight(.semibold)
                    .lineLimit(2)
                
                HStack {
                    Image(systemName: "clock")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Text("\(event.formattedStartTime)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                if event.formattedStartTime != event.formattedEndTime {
                    HStack {
                        Image(systemName: "clock.arrow.circlepath")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        
                        Text("Until \(event.formattedEndTime)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }
            
            Spacer()
        }
        .padding(.vertical, 8)
        .opacity(isDeleting ? 0.6 : 1.0)
    }
}

struct TasksListView_Previews: PreviewProvider {
    static var previews: some View {
        TasksListView(userId: "preview-user-id")
    }
}
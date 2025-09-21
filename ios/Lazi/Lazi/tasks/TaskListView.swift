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
    @State private var editingEventId: String? = nil
    @State private var updatingEventId: String? = nil
    
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
                                isEditing: editingEventId == event.event_id,
                                isUpdating: updatingEventId == event.event_id,
                                onDelete: { deleteEvent(event) },
                                onStartEdit: { startEditingEvent(event) },
                                onFinishEdit: { newName in finishEditingEvent(event, newName: newName) },
                                onCancelEdit: { cancelEditingEvent() }
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
        
    guard let url = URL(string: "https://29098e308ec4.ngrok-free.app/task_list/list_events") else {
            errorMessage = "Invalid URL"
            isLoading = false
            showingError = true
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        // Updated request body to match backend EventRequest structure
        let requestBody: [String: Any] = [
            "event_details": [
                "input_text": "list events",
                "raw_output": "None",
                "event_name": "None",
                "datetime_obj": [
                    "target_datetimes": []
                ],
                "action": "list",
                "response": "None",
                "transparency": "opaque",
                "guestsCanModify": false,
                "description": "None",
                "attendees": []
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
        
    guard let url = URL(string: "https://29098e308ec4.ngrok-free.app/scheduler/delete_event/\(event.event_id)") else {
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
    
    // MARK: - Event Editing Functions
    
    private func startEditingEvent(_ event: TaskEvent) {
        editingEventId = event.event_id
        print("Started editing event: \(event.event_name)")
    }
    
    private func cancelEditingEvent() {
        editingEventId = nil
        print("Cancelled editing")
    }
    
    private func finishEditingEvent(_ event: TaskEvent, newName: String) {
        guard !newName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            cancelEditingEvent()
            return
        }
        
        let trimmedName = newName.trimmingCharacters(in: .whitespacesAndNewlines)
        
        // Don't make API call if name hasn't changed
        if trimmedName == event.event_name {
            cancelEditingEvent()
            return
        }
        
        updatingEventId = event.event_id
        editingEventId = nil
        
        updateEventName(event, newName: trimmedName)
    }
    
    private func updateEventName(_ event: TaskEvent, newName: String) {
    guard let url = URL(string: "https://29098e308ec4.ngrok-free.app/scheduler/update_event/\(event.event_id)") else {
            errorMessage = "Invalid URL"
            showingError = true
            updatingEventId = nil
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        // Create complete responseRequest structure with updated event name
        let responseRequest: [String: Any] = [
            "user_id": userId,
            "status": "request event ID",
            "message": "Event '\(event.event_name)' selected for update from task list",
            "event_requested": [
                "event_name": newName, // Use the new name here
                "target_dates": createTargetDatetimes(from: event),
                "action": "update",
                "response": "Event updated to '\(newName)' successfully"
            ],
            "calendar_insights": [
                "matching_events": [
                    [
                        "event_name": event.event_name, // Keep original name for matching
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
        
        print("Sending update request to rename '\(event.event_name)' to '\(newName)'")
        print("- Event ID: \(event.event_id)")
        print("- Full Request: \(responseRequest)")
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: responseRequest)
        } catch {
            errorMessage = "Failed to encode request"
            showingError = true
            updatingEventId = nil
            return
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                updatingEventId = nil
                
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
                    print("Update API Response: \(responseString)")
                }
                
                do {
                    if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                       let status = json["status"] as? String {
                        if status == "success" {
                            // Update the event name in the local list
                            if let index = events.firstIndex(where: { $0.event_id == event.event_id }) {
                                let updatedEvent = TaskEvent(
                                    event_name: newName,
                                    start_time: event.start_time,
                                    end_time: event.end_time,
                                    event_id: event.event_id
                                )
                                events[index] = updatedEvent
                                print("Successfully updated event name to: \(newName)")
                            }
                        } else {
                            errorMessage = json["message"] as? String ?? "Failed to update event"
                            showingError = true
                            print("Update failed: \(errorMessage)")
                        }
                    } else {
                        errorMessage = "Invalid response format"
                        showingError = true
                        print("Invalid response format from update API")
                    }
                } catch {
                    errorMessage = "Failed to parse response: \(error.localizedDescription)"
                    showingError = true
                    print("Failed to parse update response: \(error)")
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
    let isEditing: Bool
    let isUpdating: Bool
    let onDelete: () -> Void
    let onStartEdit: () -> Void
    let onFinishEdit: (String) -> Void
    let onCancelEdit: () -> Void
    
    @State private var editingText: String = ""
    @FocusState private var isTextFieldFocused: Bool
    
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
            .disabled(isDeleting || isEditing || isUpdating)
            
            // Event details
            VStack(alignment: .leading, spacing: 4) {
                // Event name - editable when in editing mode
                if isEditing {
                    HStack {
                        TextField("Event name", text: $editingText)
                            .font(.headline)
                            .fontWeight(.semibold)
                            .focused($isTextFieldFocused)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                            .onSubmit {
                                onFinishEdit(editingText)
                            }
                        
                        // Cancel button
                        Button("Cancel") {
                            onCancelEdit()
                        }
                        .font(.caption)
                        .foregroundColor(.secondary)
                        
                        // Save button
                        Button("Save") {
                            onFinishEdit(editingText)
                        }
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundColor(.blue)
                        .disabled(editingText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                } else {
                    HStack {
                        // Event name with tap to edit
                        Button(action: {
                            editingText = event.event_name
                            onStartEdit()
                        }) {
                            HStack {
                                Text(event.event_name)
                                    .font(.headline)
                                    .fontWeight(.semibold)
                                    .lineLimit(2)
                                    .foregroundColor(.primary)
                                    .multilineTextAlignment(.leading)
                                
                                if isUpdating {
                                    ProgressView()
                                        .scaleEffect(0.8)
                                } else {
                                    Image(systemName: "pencil")
                                        .font(.caption)
                                        .foregroundColor(.blue)
                                }
                            }
                        }
                        .disabled(isDeleting || isUpdating)
                        .buttonStyle(PlainButtonStyle())
                        
                        Spacer()
                    }
                }
                
                // Time information (always shown)
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
        .opacity((isDeleting || isUpdating) ? 0.6 : 1.0)
        .onChange(of: isEditing) { isNowEditing in
            if isNowEditing {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                    isTextFieldFocused = true
                }
            }
        }
    }
}

struct TasksListView_Previews: PreviewProvider {
    static var previews: some View {
        TasksListView(userId: "preview-user-id")
    }
}
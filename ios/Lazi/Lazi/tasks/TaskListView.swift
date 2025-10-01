import SwiftUI
import AVFoundation 
import Speech

struct TaskEvent: Codable, Identifiable {
    let event_name: String
    let start: String
    let end: String
    let event_id: String
    
    // Use event_id as the identifier for Identifiable protocol
    var id: String { event_id }
    
    // Custom coding keys to match the API response
    private enum CodingKeys: String, CodingKey {
        case event_name
        case start
        case end
        case event_id
    }
    
    var formattedStartTime: String {
        return formatDateTime(start)
    }
    
    var formattedEndTime: String {
        return formatDateTime(end)
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

// MARK: - Supporting Data Models
// Models moved to SharedModels.swift to avoid redeclaration

struct TasksListView: View {
    let userId: String
    @State private var events: [TaskEvent] = []
    @State private var isLoading = false
    @State private var errorMessage = ""
    @State private var showingError = false
    @State private var deletingEventId: String? = nil
    @State private var editingEventId: String? = nil
    @State private var updatingEventId: String? = nil
    @State private var targetStart: String? = nil // New state for calendar date filtering
    @State private var targetEnd: String? = nil // New state for calendar date filtering end
    
    // Text input functionality states
    @State private var userInput: String = ""
    @State private var isProcessing = false
    @State private var showEventSelection = false
    @State private var selectableEvents: [SelectableEvent] = []
    @State private var pendingAction: String = ""
    @State private var currentResponseRequest: ResponseRequest? = nil
    @State private var storedResponseRequests: [ResponseRequest] = []
    
    // Audio functionality
    @State private var audioEnabled = false
    private let speechSynthesizer = AVSpeechSynthesizer()
    
    // Speech recognition properties
    @State private var isRecording = false
    @State private var speechRecognizer = SFSpeechRecognizer()
    @State private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    @State private var recognitionTask: SFSpeechRecognitionTask?
    @State private var audioEngine = AVAudioEngine()
    
    // MARK: - Computed Properties for View Components
    
    private var eventsList: some View {
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
                .listRowBackground(Color.clear)
                .listRowSeparator(.hidden)
            }
        }
        .listStyle(PlainListStyle())
        .background(Color.clear)
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.white, lineWidth: 1)
        )
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.black.opacity(0.05)) // Match calendar background
        )
        .cornerRadius(12)
        .padding(.horizontal, 20) // Match calendar horizontal padding
        .refreshable {
            fetchEvents(targetStart: targetStart, targetEnd: targetEnd)
        }
    }
    
    private var inputSection: some View {
        VStack(spacing: 12) {
            HStack {
                TextField("Enter task or event command...", text: $userInput)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                
                Button(action: isRecording ? stopRecording : startRecording) {
                    Image(systemName: isRecording ? "stop.circle.fill" : "mic.circle")
                        .foregroundColor(isRecording ? .red : .blue)
                        .font(.title2)
                }
                .frame(width: 30, height: 30)
                .disabled(isProcessing)
                .background(isRecording ? Color.red.opacity(0.1) : Color.clear)
                .clipShape(Circle())
                
                Button(action: sendToAPI) {
                    if isProcessing {
                        ProgressView()
                            .scaleEffect(0.8)
                    } else {
                        Image(systemName: "paperplane.fill")
                            .foregroundColor(.blue)
                    }
                }
                .frame(width: 30, height: 30)
                .disabled(userInput.isEmpty || isProcessing)
            }
            
            HStack {
                Text("Audio")
                    .foregroundColor(.secondary)
                
                Toggle("", isOn: $audioEnabled)
                    .scaleEffect(0.8)
                
                Spacer()
            }
        }
        .padding(.horizontal, 20) // Match calendar and events list padding
        .padding(.vertical, 16)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.black.opacity(0.05)) // Match other components
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.white.opacity(0.8), lineWidth: 1) // Match other components
        )
        .padding(.horizontal, 20) // Outer padding to match other components
        .padding(.bottom, 8)
    }
    
    private var calendarSection: some View {
        TaskCalendarView(userId: userId) { selectedDate, endDate in
            targetStart = selectedDate
            targetEnd = endDate
            fetchEvents(targetStart: selectedDate, targetEnd: endDate)
        }
        .padding(.horizontal, 20) // Match the horizontal padding of eventsList
        .padding(.top, 8)
        .padding(.bottom, 16) // Add bottom padding to separate from events list
    }
    
    private var loadingView: some View {
        VStack(spacing: 20) {
            ProgressView("Loading events...")
            Text("Fetching your tasks and events")
                .foregroundColor(.secondary)
                .font(.caption)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
    
    private var emptyStateView: some View {
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
                fetchEvents(targetStart: targetStart, targetEnd: targetEnd)
            }
            .buttonStyle(.bordered)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
    
    private var mainContent: some View {
        VStack(spacing: 0) {
            calendarSection
            
            if isLoading {
                loadingView
            } else if events.isEmpty {
                emptyStateView
            } else {
                eventsList
            }
            
            Spacer(minLength: 16) // Add spacing before input section
            
            inputSection
        }
        .padding(.top, 8)
    }
    
    var body: some View {
        NavigationView {
            mainContent
        }
        .navigationTitle("Tasks & Events")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                HStack {
                    // Audio toggle slider
                    HStack(spacing: 8) {
                        Image(systemName: audioEnabled ? "speaker.wave.2" : "speaker.slash")
                            .foregroundColor(audioEnabled ? .blue : .gray)
                            .font(.caption)
                        
                        Toggle("", isOn: $audioEnabled)
                            .labelsHidden()
                            .scaleEffect(0.8)
                    }
                    
                    Button("Refresh") {
                        fetchEvents()
                    }
                    .disabled(isLoading)
                }
            }
        }
        .onAppear {
            fetchEvents()
            requestSpeechAuthorization()
        }
        .alert("Error", isPresented: $showingError) {
            Button("OK") {
                self.errorMessage = ""
            }
            Button("Retry") {
                fetchEvents(targetStart: targetStart, targetEnd: targetEnd)
            }
        } message: {
            Text(errorMessage)
        }
        .sheet(isPresented: $showEventSelection) {
            EventSelectionView(
                selectableEvents: selectableEvents,
                onEventSelected: selectEvent,
                onCancel: {
                    self.showEventSelection = false
                    self.pendingAction = ""
                    self.currentResponseRequest = nil
                    self.selectableEvents = []
                }
            )
        }
    }
    
    // MARK: - Speech Recognition Functions
    
    private func requestSpeechAuthorization() {
        SFSpeechRecognizer.requestAuthorization { authStatus in
            DispatchQueue.main.async {
                switch authStatus {
                case .authorized:
                    print("Speech recognition authorized")
                case .denied:
                    print("Speech recognition denied")
                case .restricted:
                    print("Speech recognition restricted")
                case .notDetermined:
                    print("Speech recognition not determined")
                @unknown default:
                    break
                }
            }
        }
        
        AVAudioSession.sharedInstance().requestRecordPermission { granted in
            DispatchQueue.main.async {
                if granted {
                    print("Microphone permission granted")
                } else {
                    print("Microphone permission denied")
                }
            }
        }
    }
    
    private func startRecording() {
        // Cancel any previous task
        recognitionTask?.cancel()
        recognitionTask = nil
        
        // Configure audio session
        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            print("Audio session error: \(error)")
            return
        }
        
        // Create recognition request
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest = recognitionRequest else {
            print("Unable to create recognition request")
            return
        }
        
        recognitionRequest.shouldReportPartialResults = true
        
        // Create audio input
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
            recognitionRequest.append(buffer)
        }
        
        // Start audio engine
        audioEngine.prepare()
        do {
            try audioEngine.start()
        } catch {
            print("Audio engine error: \(error)")
            return
        }
        
        // Start recognition
        recognitionTask = speechRecognizer?.recognitionTask(with: recognitionRequest) { result, error in
            if let result = result {
                DispatchQueue.main.async {
                    self.userInput = result.bestTranscription.formattedString
                }
                
                if result.isFinal {
                    DispatchQueue.main.async {
                        self.stopRecording()
                    }
                }
            }
            
            if let error = error {
                print("Recognition error: \(error)")
                DispatchQueue.main.async {
                    self.stopRecording()
                }
            }
        }
        
        self.isRecording = true
    }
    
    private func stopRecording() {
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        
        recognitionRequest?.endAudio()
        recognitionRequest = nil
        
        recognitionTask?.cancel()
        recognitionTask = nil
        
        isRecording = false
    }
    
    // MARK: - Text Input API Functions
    
    private func sendToAPI() {
        guard !userInput.isEmpty else { return }
        
        let currentInput = userInput
        userInput = ""
        isProcessing = true
        
        // Step 1: Call fetch_events to extract events from the input text
        fetchEventsFromInput(inputText: currentInput)
    }
    
    private func fetchEventsFromInput(inputText: String) {
    guard let url = URL(string: "http://192.168.1.222:8000/scheduler/fetch_events") else {
            self.errorMessage = "Invalid fetch_events URL."
            self.showingError = true
            self.isProcessing = false
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let requestBody = [
            "input_text": inputText,
            "user_id": userId
        ]
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            self.errorMessage = "Failed to encode request"
            self.showingError = true
            self.isProcessing = false
            return
        }

        print("Step 1: Calling fetch_events with input: \(inputText)")

        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    self.errorMessage = "Network error: \(error.localizedDescription)"
                    self.showingError = true
                    self.isProcessing = false
                    return
                }
                
                guard let data = data else {
                    self.errorMessage = "No data received"
                    self.showingError = true
                    self.isProcessing = false
                    return
                }
                
                do {
                    if let events = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
                        print("Fetched \(events.count) events from input")
                        self.processEventsInput(events: events)
                    } else {
                        self.errorMessage = "Invalid response format"
                        self.showingError = true
                        self.isProcessing = false
                    }
                } catch {
                    self.errorMessage = "Failed to parse response: \(error.localizedDescription)"
                    self.showingError = true
                    self.isProcessing = false
                }
            }
        }.resume()
    }
    
    private func processEventsInput(events: [[String: Any]]) {
    guard let url = URL(string: "http://192.168.1.222:8000/scheduler/process_input") else {
            self.errorMessage = "Invalid process_input URL."
            self.showingError = true
            self.isProcessing = false
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: events)
        } catch {
            self.errorMessage = "Failed to encode events"
            self.showingError = true
            self.isProcessing = false
            return
        }

        print("Step 2: Calling process_input with \(events.count) events")

        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                self.isProcessing = false
                
                if let error = error {
                    self.errorMessage = "Network error: \(error.localizedDescription)"
                    self.showingError = true
                    return
                }
                
                guard let data = data else {
                    self.errorMessage = "No data received"
                    self.showingError = true
                    return
                }
                
                do {
                    if let responseArray = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
                        print("Processing \(responseArray.count) responses")
                        
                        for responseDict in responseArray {
                            if let responseRequest = self.parseResponseRequest(from: responseDict) {
                                self.processResponseRequest(responseRequest)
                            }
                        }
                        
                        // Refresh events list after processing
                        self.fetchEvents(targetStart: self.targetStart, targetEnd: self.targetEnd)
                    } else {
                        self.errorMessage = "Invalid response format"
                        self.showingError = true
                    }
                } catch {
                    self.errorMessage = "Failed to parse response: \(error.localizedDescription)"
                    self.showingError = true
                }
            }
        }.resume()
    }
    
    private func selectEvent(_ event: SelectableEvent) {
        self.showEventSelection = false
        
        guard let responseRequest = currentResponseRequest else {
            self.errorMessage = "No pending request found."
            self.showingError = true
            return
        }
        
        if pendingAction == "delete" {
            callDeleteEvent(eventId: event.eventId, selectedEvent: event, responseRequest: responseRequest)
        } else if pendingAction == "update" {
            callUpdateEvent(eventId: event.eventId, selectedEvent: event, responseRequest: responseRequest)
        }
        
        // Clear pending state
        self.pendingAction = ""
        self.currentResponseRequest = nil
        self.selectableEvents = []
    }
    
    private func callDeleteEvent(eventId: String, selectedEvent: SelectableEvent, responseRequest: ResponseRequest) {
    guard let url = URL(string: "http://192.168.1.222:8000/scheduler/delete_event/\(eventId)") else {
            self.errorMessage = "Invalid delete URL."
            self.showingError = true
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        var requestBody: [String: Any] = [
            "user_id": userId,
            "status": responseRequest.status,
            "message": responseRequest.message,
            "event_requested": responseRequest.eventRequested
        ]
        
        if let calendarInsights = responseRequest.calendarInsights {
            requestBody["calendar_insights"] = calendarInsights
        }

        print("Sending delete request for event \(eventId)")

        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            self.errorMessage = "Failed to encode request"
            self.showingError = true
            return
        }

        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    self.errorMessage = "Delete failed: \(error.localizedDescription)"
                    self.showingError = true
                    return
                }
                
                guard let data = data else {
                    self.errorMessage = "No response from delete request"
                    self.showingError = true
                    return
                }
                
                do {
                    let _ = try JSONSerialization.jsonObject(with: data)
                    print("Event deleted successfully")
                    // Refresh the events list
                    self.fetchEvents(targetStart: self.targetStart, targetEnd: self.targetEnd)
                } catch {
                    self.errorMessage = "Failed to parse delete response: \(error.localizedDescription)"
                    self.showingError = true
                }
            }
        }.resume()
    }
    
    private func callUpdateEvent(eventId: String, selectedEvent: SelectableEvent, responseRequest: ResponseRequest) {
    guard let url = URL(string: "http://192.168.1.222:8000/scheduler/update_event/\(eventId)") else {
            self.errorMessage = "Invalid update URL."
            self.showingError = true
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        var requestBody: [String: Any] = [
            "user_id": userId,
            "status": responseRequest.status,
            "message": responseRequest.message,
            "event_requested": responseRequest.eventRequested
        ]
        
        if let calendarInsights = responseRequest.calendarInsights {
            requestBody["calendar_insights"] = calendarInsights
        }

        print("Sending update request for event \(eventId)")

        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            self.errorMessage = "Failed to encode request"
            self.showingError = true
            return
        }

        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    self.errorMessage = "Update failed: \(error.localizedDescription)"
                    self.showingError = true
                    return
                }
                
                guard let data = data else {
                    self.errorMessage = "No response from update request"
                    self.showingError = true
                    return
                }
                
                do {
                    let _ = try JSONSerialization.jsonObject(with: data)
                    print("Event updated successfully")
                    // Refresh the events list
                    self.fetchEvents(targetStart: self.targetStart, targetEnd: self.targetEnd)
                } catch {
                    self.errorMessage = "Failed to parse update response: \(error.localizedDescription)"
                    self.showingError = true
                }
            }
        }.resume()
    }
    
    private func parseResponseRequest(from json: [String: Any]) -> ResponseRequest? {
        guard let status = json["status"] as? String,
              let message = json["message"] as? String,
              let eventRequested = json["event_requested"] as? [String: Any] else {
            return nil
        }
        
        let userId = json["user_id"] as? String
        let calendarInsights = json["calendar_insights"] as? [String: Any]
        
        return ResponseRequest(
            userId: userId,
            status: status,
            message: message,
            eventRequested: eventRequested,
            calendarInsights: calendarInsights
        )
    }
    
    private func processResponseRequest(_ responseRequest: ResponseRequest) {
        // Play audio message if enabled
        if audioEnabled {
            speakMessage(responseRequest.message)
        }
        
        // Check if status indicates success (completed action)
        if responseRequest.status.lowercased() == "completed" {
            return
        }
        
        // For non-success status, check if we need to handle delete/update actions
        if let action = responseRequest.eventRequested["action"] as? String {
            if action.lowercased() == "delete" {
                handleDeleteAction(responseRequest: responseRequest)
            } else if action.lowercased() == "update" {
                handleUpdateAction(responseRequest: responseRequest)
            }
        }
    }
    
    private func handleDeleteAction(responseRequest: ResponseRequest) {
        // Extract calendar insights to show available events
        guard let calendarInsights = responseRequest.calendarInsights,
              let matchingEvents = calendarInsights["matching_events"] as? [[String: Any]] else {
            return
        }
        
        self.selectableEvents = matchingEvents.compactMap { eventDict in
            guard let eventName = eventDict["event_name"] as? String,
                  let eventId = eventDict["event_id"] as? String else {
                return nil
            }
            
            return SelectableEvent(
                eventName: eventName,
                eventId: eventId,
                startDate: eventDict["start"] as? String,
                endDate: eventDict["end"] as? String
            )
        }
        
        if !self.selectableEvents.isEmpty {
            self.pendingAction = "delete"
            self.currentResponseRequest = responseRequest
            self.showEventSelection = true
        }
    }
    
    private func handleUpdateAction(responseRequest: ResponseRequest) {
        // Extract calendar insights to show available events
        guard let calendarInsights = responseRequest.calendarInsights,
              let matchingEvents = calendarInsights["matching_events"] as? [[String: Any]] else {
            return
        }
        
        self.selectableEvents = matchingEvents.compactMap { eventDict in
            guard let eventName = eventDict["event_name"] as? String,
                  let eventId = eventDict["event_id"] as? String else {
                return nil
            }
            
            return SelectableEvent(
                eventName: eventName,
                eventId: eventId,
                startDate: eventDict["start"] as? String,
                endDate: eventDict["end"] as? String
            )
        }
        
        if !self.selectableEvents.isEmpty {
            self.pendingAction = "update"
            self.currentResponseRequest = responseRequest
            self.showEventSelection = true
        }
    }
    
    private func speakMessage(_ message: String) {
        let utterance = AVSpeechUtterance(string: message)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = 0.5
        speechSynthesizer.speak(utterance)
    }
    
    private func fetchEvents(targetStart: String? = nil, targetEnd: String? = nil) {
        self.isLoading = true
        self.errorMessage = ""

    guard let url = URL(string: "http://192.168.1.222:8000/task_list/list_events") else {
            self.errorMessage = "Invalid URL"
            self.isLoading = false
            self.showingError = true
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        // Updated request body to match backend EventRequest structure with target_start and target_end support
        var requestBody: [String: Any] = [
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
        
        // Add target_start and target_end if provided
        if let targetStart = targetStart {
            requestBody["target_start"] = targetStart
            print("Fetching events with target_start: \(targetStart)")
        }
        
        if let targetEnd = targetEnd {
            requestBody["target_end"] = targetEnd
            print("Fetching events with target_end: \(targetEnd)")
        }
        
        print("Request body: \(requestBody)")
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            self.errorMessage = "Failed to encode request"
            self.isLoading = false
            self.showingError = true
            return
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                self.isLoading = false
                
                if let error = error {
                    self.errorMessage = "Network error: \(error.localizedDescription)"
                    self.showingError = true
                    return
                }
                
                guard let data = data else {
                    self.errorMessage = "No data received"
                    self.showingError = true
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
                               let startTime = eventDict["start"] as? String,
                               let endTime = eventDict["end"] as? String,
                               let eventId = eventDict["event_id"] as? String {
                                
                                let taskEvent = TaskEvent(
                                    event_name: eventName,
                                    start: startTime,
                                    end: endTime,
                                    event_id: eventId
                                )
                                parsedEvents.append(taskEvent)
                            }
                        }
                        
                        print("Parsed \(parsedEvents.count) events successfully")
                        self.events = parsedEvents
                    } else {
                        self.errorMessage = "Invalid response format"
                        self.showingError = true
                    }
                } catch {
                    self.errorMessage = "Failed to parse response: \(error.localizedDescription)"
                    self.showingError = true
                }
            }
        }.resume()
    }
    
    private func deleteEvent(_ event: TaskEvent) {
        self.deletingEventId = event.event_id

        guard let url = URL(string: "http://192.168.1.222:8000/scheduler/delete_event/\(event.event_id)") else {
            self.errorMessage = "Invalid URL"
            self.showingError = true
            self.deletingEventId = nil
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
                        "start": event.start,
                        "end": event.end.isEmpty ? event.start : event.end,
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
        print("- Start Time: \(event.start)")
        print("- End Time: \(event.end)")
        print("- Is Event: \(determineIsEvent(from: event))")
        print("- Full Request: \(responseRequest)")
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: responseRequest)
        } catch {
            self.errorMessage = "Failed to encode request"
            self.showingError = true
            self.deletingEventId = nil
            return
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                self.deletingEventId = nil
                
                if let error = error {
                    self.errorMessage = "Delete failed: \(error.localizedDescription)"
                    self.showingError = true
                    return
                }
                
                guard let data = data else {
                    self.errorMessage = "No response from server"
                    self.showingError = true
                    return
                }
                
                // Log the raw response for debugging
                if let responseString = String(data: data, encoding: .utf8) {
                    print("Delete response: \(responseString)")
                }
                
                do {
                    let _ = try JSONSerialization.jsonObject(with: data)
                    print("Event deleted successfully")
                    // Refresh events after successful deletion
                    self.fetchEvents(targetStart: self.targetStart, targetEnd: self.targetEnd)
                } catch {
                    self.errorMessage = "Failed to parse delete response: \(error.localizedDescription)"
                    self.showingError = true
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
        guard let url = URL(string: "http://192.168.1.222:8000/scheduler/update_event/\(event.event_id)") else {
            self.errorMessage = "Invalid URL"
            self.showingError = true
            self.updatingEventId = nil
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
                        "start": event.start,
                        "end": event.end.isEmpty ? event.start : event.end,
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
            self.errorMessage = "Failed to encode request"
            self.showingError = true
            self.updatingEventId = nil
            return
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                self.updatingEventId = nil
                
                if let error = error {
                    self.errorMessage = "Update failed: \(error.localizedDescription)"
                    self.showingError = true
                    return
                }
                
                guard let data = data else {
                    self.errorMessage = "No response from server"
                    self.showingError = true
                    return
                }
                
                // Log the raw response for debugging
                if let responseString = String(data: data, encoding: .utf8) {
                    print("Update API Response: \(responseString)")
                }
                
                do {
                    let _ = try JSONSerialization.jsonObject(with: data)
                    print("Event name updated successfully")
                    // Refresh events after successful update
                    self.fetchEvents(targetStart: self.targetStart, targetEnd: self.targetEnd)
                } catch {
                    self.errorMessage = "Failed to parse update response: \(error.localizedDescription)"
                    self.showingError = true
                }
            }
        }.resume()
    }
    
    // MARK: - Helper Functions for Event Details Compilation (Updated)
    
    private func createTargetDatetimes(from event: TaskEvent) -> [[String]] {
        // Convert the event's start and end times to the format expected by the API
        let startTime = event.start
        let endTime = event.end
        
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
        let isEvent = !event.end.isEmpty && event.end != event.start
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
        .padding(.horizontal, 12)
        .background(Color.clear)
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
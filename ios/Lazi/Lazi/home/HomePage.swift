import SwiftUI
import CoreHaptics
import Speech
import AVFoundation

struct ChatBubbleView: View {
    let message: ChatMessage
    
    var body: some View {
        HStack {
            if message.isUserMessage {
                Spacer()
                Text(message.text)
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(12)
                    .frame(maxWidth: UIScreen.main.bounds.width * 0.7, alignment: .trailing)
            } else {
                Text(message.text)
                    .padding()
                    .background(Color(UIColor.systemGray5))
                    .foregroundColor(.primary)
                    .cornerRadius(12)
                    .frame(maxWidth: UIScreen.main.bounds.width * 0.7, alignment: .leading)
                Spacer()
            }
        }
    }
}

struct CalendarEvent: Codable {
    let id: String = UUID().uuidString
    let eventName: String
    let startDate: String
    let endDate: String
    let isEvent: Bool
    let eventId: String
}

struct SelectableEvent: Identifiable {
    let id = UUID()
    let eventName: String
    let eventId: String
    let startDate: String?
    let endDate: String?
}

// Response structure matching the API
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

struct EventSelectionView: View {
    let selectableEvents: [SelectableEvent]
    let onEventSelected: (SelectableEvent) -> Void
    let onCancel: () -> Void
    
    var body: some View {
        NavigationView {
            List(selectableEvents) { event in
                Button(action: {
                    onEventSelected(event)
                }) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(event.eventName)
                            .font(.headline)
                            .foregroundColor(.primary)
                        
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
                    .padding(.vertical, 4)
                }
                .buttonStyle(PlainButtonStyle())
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

struct HomePage: View {
    let userId: String
    
    @State private var userInput: String = ""
    @State private var responseMessage: String = ""
    @State private var messages: [ChatMessage] = []
    @State private var buttonTapped: Bool = false
    @State private var status: String = ""
    
    // Store all response requests from the first API call
    @State private var storedResponseRequests: [ResponseRequest] = []
    
    // Event selection states - storing complete responseRequest data
    @State private var showEventSelection = false
    @State private var selectableEvents: [SelectableEvent] = []
    @State private var pendingAction: String = ""
    @State private var currentResponseRequest: ResponseRequest? = nil
    
    // Speech recognition properties
    @State private var isRecording = false
    @State private var speechRecognizer = SFSpeechRecognizer()
    @State private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    @State private var recognitionTask: SFSpeechRecognitionTask?
    @State private var audioEngine = AVAudioEngine()
    
    // Sidebar state
    @State private var showSidebar = false

    var body: some View {
        ZStack {
            NavigationView {
                VStack(spacing: 16) {
                    // Header
                    VStack(spacing: 8) {
                        Text("Lazi")
                            .font(.largeTitle)
                            .fontWeight(.bold)
                            .foregroundColor(.primary)
                        
                        Text("Your Personal Assistant")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    .padding(.top, 16)
                    
                    // Chat interface
                    chatInterface
                    
                    // Input section
                    inputSection
                }
                .navigationBarHidden(true)
                .padding()
                .onAppear {
                    requestSpeechAuthorization()
                }
                .sheet(isPresented: $showEventSelection) {
                    EventSelectionView(
                        selectableEvents: selectableEvents,
                        onEventSelected: { event in
                            selectEvent(event)
                        },
                        onCancel: {
                            showEventSelection = false
                            selectableEvents = []
                            pendingAction = ""
                            currentResponseRequest = nil
                        }
                    )
                }
            }
            
            // Sidebar overlay
            if showSidebar {
                Color.black.opacity(0.3)
                    .ignoresSafeArea()
                    .onTapGesture {
                        withAnimation(.easeInOut(duration: 0.3)) {
                            showSidebar = false
                        }
                    }
                
                HStack {
                    SidebarView(isPresented: $showSidebar, userId: userId)
                        .transition(.move(edge: .leading))
                    
                    Spacer()
                }
            }
            
            // Top-left sidebar button
            VStack {
                HStack {
                    Button(action: {
                        withAnimation(.easeInOut(duration: 0.3)) {
                            showSidebar.toggle()
                        }
                    }) {
                        Image(systemName: "line.horizontal.3")
                            .font(.title2)
                            .foregroundColor(.primary)
                            .padding(12)
                            .background(Color(UIColor.systemBackground))
                            .clipShape(Circle())
                            .shadow(color: .black.opacity(0.1), radius: 3, x: 0, y: 2)
                    }
                    .padding(.leading, 16)
                    .padding(.top, 16)
                    
                    Spacer()
                }
                
                Spacer()
            }
        }
    }

    // MARK: - View Components
    
    var chatInterface: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 12) {
                ForEach(messages.indices, id: \.self) { index in
                    ChatBubbleView(message: messages[index])
                }
            }
            .padding(.horizontal)
        }
        .frame(maxHeight: 400)
        .background(Color(UIColor.systemGray6))
        .cornerRadius(12)
    }
    
    var inputSection: some View {
        VStack(spacing: 12) {
            HStack {
                TextEditor(text: $userInput)
                    .frame(minHeight: 40, maxHeight: 120)
                    .padding(8)
                    .background(Color(UIColor.systemGray5))
                    .cornerRadius(8)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.gray.opacity(0.3), lineWidth: 1)
                    )
                
                VStack(spacing: 8) {
                    Button(action: {
                        if isRecording {
                            stopRecording()
                        } else {
                            startRecording()
                        }
                    }) {
                        Image(systemName: isRecording ? "mic.fill" : "mic")
                            .font(.title2)
                            .foregroundColor(isRecording ? .red : .blue)
                            .frame(width: 44, height: 44)
                            .background(Color(UIColor.systemGray5))
                            .clipShape(Circle())
                    }
                    
                    Button(action: sendToAPI) {
                        Image(systemName: "paperplane.fill")
                            .font(.title2)
                            .foregroundColor(.white)
                            .frame(width: 44, height: 44)
                            .background(Color.blue)
                            .clipShape(Circle())
                    }
                    .disabled(userInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    
                    Button(action: clearChatMessages) {
                        Image(systemName: "trash")
                            .font(.title2)
                            .foregroundColor(.white)
                            .frame(width: 44, height: 44)
                            .background(Color.red)
                            .clipShape(Circle())
                    }
                    .disabled(messages.isEmpty)
                }
            }
            
            if !status.isEmpty {
                Text(status)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding(.horizontal)
            }
        }
    }

    // MARK: - Helper Functions
    
    func clearChatMessages() {
        messages.removeAll()
        // Clear all stored state
        storedResponseRequests.removeAll()
        showEventSelection = false
        selectableEvents = []
        pendingAction = ""
        currentResponseRequest = nil
    }
    
    func formatDate(_ dateString: String) -> String {
        let formatter = ISO8601DateFormatter()
        if let date = formatter.date(from: dateString) {
            let displayFormatter = DateFormatter()
            displayFormatter.dateStyle = .short
            displayFormatter.timeStyle = .short
            return displayFormatter.string(from: date)
        }
        return dateString
    }
    
    func selectEvent(_ event: SelectableEvent) {
        showEventSelection = false
        
        guard let responseRequest = currentResponseRequest else {
            messages.append(ChatMessage(text: "Error: No pending request found.", isUserMessage: false))
            return
        }
        
        if pendingAction == "delete" {
            callDeleteEvent(eventId: event.eventId, selectedEvent: event, responseRequest: responseRequest)
        } else if pendingAction == "update" {
            callUpdateEvent(eventId: event.eventId, selectedEvent: event, responseRequest: responseRequest)
        }
        
        // Clear pending state
        pendingAction = ""
        currentResponseRequest = nil
        selectableEvents = []
    }

    // MARK: - Handle Delete Action (Updated to use ResponseRequest structure)
    func handleDeleteAction(responseRequest: ResponseRequest) {
        self.messages.append(ChatMessage(text: responseRequest.message, isUserMessage: false))
        
        // Extract calendar insights to show available events
        if let calendarInsights = responseRequest.calendarInsights,
           let matchingEvents = calendarInsights["matching_events"] as? [[String: Any]] {
            
            // Show available events to user
            if matchingEvents.isEmpty {
                self.messages.append(ChatMessage(text: "No matching events found to delete.", isUserMessage: false))
            } else {
                // Convert to SelectableEvent objects with proper field mapping
                selectableEvents = matchingEvents.compactMap { event in
                    guard let eventName = event["event_name"] as? String,
                          let eventId = event["event_id"] as? String else {
                        print("Missing event_name or event_id in: \(event)")
                        return nil
                    }
                    
                    let startDate = event["start"] as? String
                    let endDate = event["end"] as? String
                    
                    return SelectableEvent(
                        eventName: eventName,
                        eventId: eventId,
                        startDate: startDate,
                        endDate: endDate
                    )
                }
                
                // Store pending action and current responseRequest
                pendingAction = "delete"
                currentResponseRequest = responseRequest
                
                // Show selection UI
                showEventSelection = true
                
                // Also show a message about the selection
                self.messages.append(ChatMessage(text: "Found \(selectableEvents.count) matching event(s). Please select one to delete.", isUserMessage: false))
            }
        }
    }

    // MARK: - Handle Update Action (Updated to use ResponseRequest structure)
    func handleUpdateAction(responseRequest: ResponseRequest) {
        self.messages.append(ChatMessage(text: responseRequest.message, isUserMessage: false))
        
        // Extract calendar insights to show available events
        if let calendarInsights = responseRequest.calendarInsights,
           let matchingEvents = calendarInsights["matching_events"] as? [[String: Any]] {
            
            // Show available events to user
            if matchingEvents.isEmpty {
                self.messages.append(ChatMessage(text: "No matching events found to update.", isUserMessage: false))
            } else {
                // Convert to SelectableEvent objects with proper field mapping
                selectableEvents = matchingEvents.compactMap { event in
                    guard let eventName = event["event_name"] as? String,
                          let eventId = event["event_id"] as? String else {
                        print("Missing event_name or event_id in: \(event)")
                        return nil
                    }
                    
                    let startDate = event["start"] as? String
                    let endDate = event["end"] as? String
                    
                    return SelectableEvent(
                        eventName: eventName,
                        eventId: eventId,
                        startDate: startDate,
                        endDate: endDate
                    )
                }
                
                // Store pending action and current responseRequest
                pendingAction = "update"
                currentResponseRequest = responseRequest
                
                // Show selection UI
                showEventSelection = true
                
                // Also show a message about the selection
                self.messages.append(ChatMessage(text: "Found \(selectableEvents.count) matching event(s). Please select one to update.", isUserMessage: false))
            }
        }
    }

    // MARK: - Speech Recognition Functions (keeping your existing implementation)
    
    func requestSpeechAuthorization() {
        requestSpeechPermission()
    }
    
    func requestSpeechPermission() {
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
                    print("Unknown speech recognition status")
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
    
    func startRecording() {
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
        
        isRecording = true
    }
    
    func stopRecording() {
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        
        recognitionRequest?.endAudio()
        recognitionRequest = nil
        
        recognitionTask?.cancel()
        recognitionTask = nil
        
        isRecording = false
    }

    // MARK: - API Functions (Updated for new two-step process)
    
    func sendToAPI() {
        guard !userInput.isEmpty else {
            responseMessage = "Please enter a task."
            return
        }
        
        // Add user message (always true for user input)
        messages.append(ChatMessage(text: userInput, isUserMessage: true))
        let currentInput = userInput
        userInput = ""  // Clear the input field only after sending

        // Step 1: Call fetch_events to extract events from the input text
        fetchEventsFromInput(inputText: currentInput)
    }
    
    // MARK: - Step 1: Fetch Events from Input Text
    func fetchEventsFromInput(inputText: String) {
    guard let url = URL(string: "https://29098e308ec4.ngrok-free.app/scheduler/fetch_events") else {
            messages.append(ChatMessage(text: "Invalid fetch_events URL.", isUserMessage: false))
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        // Create request body for fetch_events
        let requestBody = [
            "input_text": inputText,
            "user_id": userId
        ]
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            messages.append(ChatMessage(text: "Failed to encode fetch_events request.", isUserMessage: false))
            return
        }

        print("Step 1: Calling fetch_events with input: \(inputText)")

        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    self.messages.append(ChatMessage(text: "Fetch events error: \(error.localizedDescription)", isUserMessage: false))
                    return
                }

                guard let data = data else {
                    self.messages.append(ChatMessage(text: "No data received from fetch_events.", isUserMessage: false))
                    return
                }

                do {
                    // Parse the events array from fetch_events response
                    if let eventsArray = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
                        print("Step 1 completed: Received \(eventsArray.count) events from fetch_events")
                        print("Events data: \(eventsArray)")
                        
                        // Step 2: Pass the events to process_input
                        self.processEventsInput(events: eventsArray)
                    } else {
                        self.messages.append(ChatMessage(text: "Invalid response format from fetch_events.", isUserMessage: false))
                    }
                } catch {
                    let rawResponse = String(data: data, encoding: .utf8) ?? "Failed to parse fetch_events response"
                    self.messages.append(ChatMessage(text: "Fetch events parsing error: \(rawResponse)", isUserMessage: false))
                }
            }
        }
        task.resume()
    }
    
    // MARK: - Step 2: Process Events Input
    func processEventsInput(events: [[String: Any]]) {
    guard let url = URL(string: "https://29098e308ec4.ngrok-free.app/scheduler/process_input") else {
            messages.append(ChatMessage(text: "Invalid process_input URL.", isUserMessage: false))
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        // Pass the events array directly to process_input
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: events)
        } catch {
            messages.append(ChatMessage(text: "Failed to encode process_input request.", isUserMessage: false))
            return
        }

        print("Step 2: Calling process_input with \(events.count) events")

        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    self.messages.append(ChatMessage(text: "Process input error: \(error.localizedDescription)", isUserMessage: false))
                    return
                }

                guard let data = data else {
                    self.messages.append(ChatMessage(text: "No data received from process_input.", isUserMessage: false))
                    return
                }

                do {
                    // Parse as array of ResponseRequest objects (same as before)
                    if let jsonArray = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
                        // Convert raw JSON to ResponseRequest objects
                        self.storedResponseRequests = jsonArray.compactMap { jsonObj in
                            self.parseResponseRequest(from: jsonObj)
                        }
                        
                        print("Step 2 completed: Received \(self.storedResponseRequests.count) response requests")
                        
                        // Process each stored response request (same as before)
                        for responseRequest in self.storedResponseRequests {
                            self.processResponseRequest(responseRequest)
                        }
                    } else {
                        // Fallback for single response object
                        if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                            if let responseRequest = self.parseResponseRequest(from: json) {
                                self.storedResponseRequests = [responseRequest]
                                self.processResponseRequest(responseRequest)
                            } else if let message = json["message"] as? String {
                                self.messages.append(ChatMessage(text: message, isUserMessage: false))
                            }
                        }
                    }
                } catch {
                    let rawResponse = String(data: data, encoding: .utf8) ?? "Failed to parse process_input response"
                    self.messages.append(ChatMessage(text: "Process input parsing error: \(rawResponse)", isUserMessage: false))
                }
            }
        }
        task.resume()
    }

    // MARK: - Call Delete Event API (Updated URL to match new API structure)
    func callDeleteEvent(eventId: String, selectedEvent: SelectableEvent, responseRequest: ResponseRequest) {
    guard let url = URL(string: "https://29098e308ec4.ngrok-free.app/scheduler/delete_event/\(eventId)") else {
            messages.append(ChatMessage(text: "Invalid delete URL.", isUserMessage: false))
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        // Create the request body matching the API expectations
        var requestBody: [String: Any] = [
            "user_id": userId,
            "event_requested": responseRequest.eventRequested
        ]
        
        if let calendarInsights = responseRequest.calendarInsights {
            requestBody["calendar_insights"] = calendarInsights
        }

        print("Sending delete request for event \(eventId): \(requestBody)")

        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            messages.append(ChatMessage(text: "Failed to encode delete request.", isUserMessage: false))
            return
        }

        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    self.messages.append(ChatMessage(text: "Delete error: \(error.localizedDescription)", isUserMessage: false))
                    return
                }

                guard let data = data else {
                    self.messages.append(ChatMessage(text: "No response from delete.", isUserMessage: false))
                    return
                }

                do {
                    if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                       let status = json["status"] as? String {
                        if status == "success" {
                            if let message = json["message"] as? String {
                                self.messages.append(ChatMessage(text: "✅ \(message)", isUserMessage: false))
                            } else {
                                self.messages.append(ChatMessage(text: "✅ Event deleted successfully", isUserMessage: false))
                            }
                        } else {
                            let errorMsg = json["message"] as? String ?? "Delete failed"
                            self.messages.append(ChatMessage(text: "❌ \(errorMsg)", isUserMessage: false))
                        }
                    }
                } catch {
                    let rawResponse = String(data: data, encoding: .utf8) ?? "Delete completed"
                    self.messages.append(ChatMessage(text: rawResponse, isUserMessage: false))
                }
            }
        }
        task.resume()
    }

    // MARK: - Call Update Event API (Updated URL to match new API structure)
    func callUpdateEvent(eventId: String, selectedEvent: SelectableEvent, responseRequest: ResponseRequest) {
    guard let url = URL(string: "https://29098e308ec4.ngrok-free.app/scheduler/update_event/\(eventId)") else {
            messages.append(ChatMessage(text: "Invalid update URL.", isUserMessage: false))
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        // Create the request body matching the API expectations
        var requestBody: [String: Any] = [
            "user_id": userId,
            "event_requested": responseRequest.eventRequested
        ]
        
        if let calendarInsights = responseRequest.calendarInsights {
            requestBody["calendar_insights"] = calendarInsights
        }

        print("Sending update request for event \(eventId): \(requestBody)")

        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            messages.append(ChatMessage(text: "Failed to encode update request.", isUserMessage: false))
            return
        }

        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    self.messages.append(ChatMessage(text: "Update error: \(error.localizedDescription)", isUserMessage: false))
                    return
                }

                guard let data = data else {
                    self.messages.append(ChatMessage(text: "No response from update.", isUserMessage: false))
                    return
                }

                do {
                    if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                       let status = json["status"] as? String {
                        if status == "success" {
                            if let message = json["message"] as? String {
                                self.messages.append(ChatMessage(text: "🔄 \(message)", isUserMessage: false))
                            } else {
                                self.messages.append(ChatMessage(text: "🔄 Event updated successfully", isUserMessage: false))
                            }
                        } else {
                            let errorMsg = json["message"] as? String ?? "Update failed"
                            self.messages.append(ChatMessage(text: "❌ \(errorMsg)", isUserMessage: false))
                        }
                    }
                } catch {
                    let rawResponse = String(data: data, encoding: .utf8) ?? "Update completed"
                    self.messages.append(ChatMessage(text: rawResponse, isUserMessage: false))
                }
            }
        }
        task.resume()
    }
    
    // MARK: - Helper function to parse ResponseRequest from JSON
    func parseResponseRequest(from json: [String: Any]) -> ResponseRequest? {
        guard let status = json["status"] as? String,
              let message = json["message"] as? String,
              let eventRequested = json["event_requested"] as? [String: Any] else {
            print("Failed to parse required fields from response: \(json)")
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
    
    // MARK: - Process individual ResponseRequest based on status and action
    func processResponseRequest(_ responseRequest: ResponseRequest) {
        // Check if status indicates success (completed action)
        if responseRequest.status.lowercased() == "completed" {
            // For completed actions, just show the message
            self.messages.append(ChatMessage(text: responseRequest.message, isUserMessage: false))
            return
        }
        
        // For non-success status, check if we need to handle delete/update actions
        if let action = responseRequest.eventRequested["action"] as? String {
            switch action.lowercased() {
            case "delete":
                handleDeleteAction(responseRequest: responseRequest)
            case "update":
                handleUpdateAction(responseRequest: responseRequest)
            default:
                // For other actions or unknown statuses, just show the message
                self.messages.append(ChatMessage(text: responseRequest.message, isUserMessage: false))
            }
        } else {
            // If no action found, just show the message
            self.messages.append(ChatMessage(text: responseRequest.message, isUserMessage: false))
        }
    }
}

struct HomePage_Previews: PreviewProvider {
    static var previews: some View {
        HomePage(userId: "preview-user-id")
    }
}

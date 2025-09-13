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
    let userId: String  // Add this parameter
    
    @State private var userInput: String = ""
    @State private var responseMessage: String = ""
    @State private var messages: [ChatMessage] = []
    @State private var buttonTapped: Bool = false
    @State private var status: String = ""
    
    // Event selection states
    @State private var showEventSelection = false
    @State private var selectableEvents: [SelectableEvent] = []
    @State private var pendingAction: String = ""
    @State private var pendingEventDetails: [String: Any] = [:]
    @State private var pendingCalendarInsights: [String: Any] = [:]
    
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
                            pendingEventDetails = [:]
                            pendingCalendarInsights = [:]
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
        // Also clear any pending event selection state
        showEventSelection = false
        selectableEvents = []
        pendingAction = ""
        pendingEventDetails = [:]
        pendingCalendarInsights = [:]
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
        
        if pendingAction == "delete" {
            callDeleteEvent(eventId: event.eventId, eventDetails: pendingEventDetails)
        } else if pendingAction == "update" {
            callUpdateEvent(eventId: event.eventId, eventDetails: pendingEventDetails, calendarInsights: pendingCalendarInsights)
        }
        
        // Clear pending state
        pendingAction = ""
        pendingEventDetails = [:]
        pendingCalendarInsights = [:]
        selectableEvents = []
    }

    // MARK: - Handle Delete Action (Fixed)
    func handleDeleteAction(responseObj: [String: Any], message: String) {
        self.messages.append(ChatMessage(text: message, isUserMessage: false))
        
        // Extract calendar insights to show available events
        if let calendarInsights = responseObj["calendar_insights"] as? [String: Any],
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
                
                // Store pending action details
                pendingAction = "delete"
                pendingEventDetails = responseObj["event_requested"] as? [String: Any] ?? [:]
                pendingCalendarInsights = calendarInsights
                
                // Show selection UI
                showEventSelection = true
                
                // Also show a message about the selection
                self.messages.append(ChatMessage(text: "Found \(selectableEvents.count) matching event(s). Please select one to delete.", isUserMessage: false))
            }
        }
    }

    // MARK: - Handle Update Action (Fixed)
    func handleUpdateAction(responseObj: [String: Any], message: String) {
        self.messages.append(ChatMessage(text: message, isUserMessage: false))
        
        // Extract calendar insights to show available events
        if let calendarInsights = responseObj["calendar_insights"] as? [String: Any],
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
                
                // Store pending action details
                pendingAction = "update"
                pendingEventDetails = responseObj["event_requested"] as? [String: Any] ?? [:]
                pendingCalendarInsights = calendarInsights
                
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

    // MARK: - API Functions (keeping your existing implementation with minor updates)
    
    func sendToAPI() {
        guard !userInput.isEmpty else {
            responseMessage = "Please enter a task."
            return
        }
        
        // Add user message (always true for user input)
        messages.append(ChatMessage(text: userInput, isUserMessage: true))
        let currentInput = userInput
        userInput = ""  // Clear the input field only after sending

        guard let url = URL(string: "https://f8cb321bfd70.ngrok-free.app/scheduler/process_input") else {
            messages.append(ChatMessage(text: "Invalid URL.", isUserMessage: false))
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        // Include user_id in the request body
        let requestBody = [
            "input_text": currentInput,
            "user_id": userId
        ]
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            messages.append(ChatMessage(text: "Failed to encode request.", isUserMessage: false))
            return
        }

        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    self.messages.append(ChatMessage(text: "Error: \(error.localizedDescription)", isUserMessage: false))
                    return
                }

                guard let data = data else {
                    self.messages.append(ChatMessage(text: "No data received.", isUserMessage: false))
                    return
                }

                do {
                    // Parse as array of ResponseRequest objects
                    if let jsonArray = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
                        for responseObj in jsonArray {
                            if let status = responseObj["status"] as? String,
                               let message = responseObj["message"] as? String,
                               let eventRequested = responseObj["event_requested"] as? [String: Any],
                               let action = eventRequested["action"] as? String {
                                
                                // Handle different actions
                                switch action.lowercased() {
                                case "add":
                                    // For add actions, just show the message
                                    self.messages.append(ChatMessage(text: message, isUserMessage: false))
                                    
                                case "delete":
                                    // For delete actions, show available events and call delete endpoint
                                    self.handleDeleteAction(responseObj: responseObj, message: message)
                                    
                                case "update":
                                    // For update actions, show available events and call update endpoint
                                    self.handleUpdateAction(responseObj: responseObj, message: message)
                                    
                                default:
                                    self.messages.append(ChatMessage(text: message, isUserMessage: false))
                                }
                            }
                        }
                    } else {
                        // Fallback for single response object
                        if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                           let message = json["message"] as? String {
                            self.messages.append(ChatMessage(text: message, isUserMessage: false))
                        }
                    }
                } catch {
                    let rawResponse = String(data: data, encoding: .utf8) ?? "Failed to parse response"
                    self.messages.append(ChatMessage(text: rawResponse, isUserMessage: false))
                }
            }
        }
        task.resume()
    }

    // MARK: - Call Delete Event API
    func callDeleteEvent(eventId: String, eventDetails: [String: Any]) {
        guard let url = URL(string: "https://f8cb321bfd70.ngrok-free.app/scheduler/delete_event/\(eventId)") else {
            messages.append(ChatMessage(text: "Invalid delete URL.", isUserMessage: false))
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        // Include user_id in the request body
        let requestBody: [String: Any] = [
            "event_details": eventDetails,
            "user_id": userId
        ]

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
                       let message = json["message"] as? String {
                        self.messages.append(ChatMessage(text: "✅ \(message)", isUserMessage: false))
                    }
                } catch {
                    let rawResponse = String(data: data, encoding: .utf8) ?? "Delete completed"
                    self.messages.append(ChatMessage(text: rawResponse, isUserMessage: false))
                }
            }
        }
        task.resume()
    }

    // MARK: - Call Update Event API
    func callUpdateEvent(eventId: String, eventDetails: [String: Any], calendarInsights: [String: Any]) {
        guard let url = URL(string: "https://f8cb321bfd70.ngrok-free.app/scheduler/update_event/\(eventId)") else {
            messages.append(ChatMessage(text: "Invalid update URL.", isUserMessage: false))
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        // Combine event details, calendar insights, and user_id for the request body
        let requestBody: [String: Any] = [
            "event_details": eventDetails,
            "calendar_insights": calendarInsights,
            "user_id": userId
        ]

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
                       let message = json["message"] as? String {
                        self.messages.append(ChatMessage(text: "🔄 \(message)", isUserMessage: false))
                    }
                } catch {
                    let rawResponse = String(data: data, encoding: .utf8) ?? "Update completed"
                    self.messages.append(ChatMessage(text: rawResponse, isUserMessage: false))
                }
            }
        }
        task.resume()
    }
}

struct HomePage_Previews: PreviewProvider {
    static var previews: some View {
        HomePage(userId: "preview-user-id")
    }
}

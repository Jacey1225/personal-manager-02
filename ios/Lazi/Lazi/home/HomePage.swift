import SwiftUI
import CoreHaptics
import Speech
import AVFoundation

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

struct HomePage: View {
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

    var body: some View {
        ZStack {
            VStack(spacing: 0) {
                // Header
                VStack {
                    Text("Lazi")
                        .font(.system(size: 28, weight: .bold, design: .rounded))
                        .foregroundColor(.purple.opacity(0.67))
                        .padding(.top, 10)
                        .padding(.bottom, 15)
                }

                // Chat messages - takes up available space
                ChatMessagesView(messages: messages)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                
                // Input section - always at bottom
                HStack {
                    ZStack(alignment: .trailing) {
                        TextField("Give me a task!", text: $userInput)
                            .font(.system(size: 16, weight: .medium, design: .rounded))
                            .foregroundColor(.black)
                            .padding(10)
                            .padding(.trailing, 40)
                            .background(Color.white)
                            .cornerRadius(20)
                            .overlay(
                                RoundedRectangle(cornerRadius: 20)
                                    .stroke(Color.purple.opacity(0.67), lineWidth: 2)
                            )
                            .padding(.horizontal, 20)
                        
                        Button(action: {
                            sendToAPI()
                        }) {
                            Image(systemName: "paperplane.fill")
                                .foregroundColor(.purple.opacity(0.67))
                                .padding(10)
                        }     
                        .padding(.trailing, 25)           
                    }

                    Button(action: {
                        let haptic = UIImpactFeedbackGenerator(style: .light)
                        haptic.impactOccurred()
                        
                        if isRecording {
                            stopRecording()
                        } else {
                            startRecording()
                        }
                    }) {
                        Image(systemName: isRecording ? "stop.circle.fill" : "speaker.wave.2.fill")
                            .foregroundColor(isRecording ? .red : .purple.opacity(0.67))
                            .padding(10)
                            .background(Color.white)
                            .clipShape(Circle())
                            .overlay(
                                Circle()
                                    .stroke(isRecording ? .red : Color.purple.opacity(0.67), lineWidth: 2)
                            )
                            .scaleEffect(isRecording ? 1.1 : 1.0)
                            .animation(.easeInOut(duration: 0.2), value: isRecording)
                    }
                }
                .padding()
                .background(Color.white)
            }
            .navigationBarHidden(true)
            .background(Color.white)
            .onAppear {
                requestSpeechPermission()
            }
            
            // Event Selection Overlay
            if showEventSelection {
                Color.black.opacity(0.3)
                    .ignoresSafeArea()
                    .onTapGesture {
                        showEventSelection = false
                    }
                
                VStack(spacing: 20) {
                    Text(pendingAction == "delete" ? "Select Event to Delete" : "Select Event to Update")
                        .font(.headline)
                        .foregroundColor(.black)  // Changed to black
                        .padding()
                    
                    ScrollView {
                        LazyVStack(spacing: 12) {
                            ForEach(selectableEvents) { event in
                                Button(action: {
                                    selectEvent(event)
                                }) {
                                    VStack(alignment: .leading, spacing: 8) {
                                        Text(event.eventName)
                                            .font(.system(size: 16, weight: .semibold))
                                            .foregroundColor(.black)  // Changed to black
                                        
                                        if let startDate = event.startDate {
                                            Text("Start: \(formatDate(startDate))")
                                                .font(.caption)
                                                .foregroundColor(.black)  // Changed to black
                                        }
                                        
                                        if let endDate = event.endDate {
                                            Text("End: \(formatDate(endDate))")
                                                .font(.caption)
                                                .foregroundColor(.black)  // Changed to black
                                        }
                                    }
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .padding()
                                    .background(Color.white)
                                    .cornerRadius(12)
                                    .shadow(color: .gray.opacity(0.2), radius: 2, x: 0, y: 1)
                                }
                                .buttonStyle(PlainButtonStyle())
                            }
                        }
                        .padding(.horizontal)
                    }
                    .frame(maxHeight: 400)
                    
                    Button("Cancel") {
                        showEventSelection = false
                    }
                    .foregroundColor(.black)  // Changed to black
                    .padding()
                    .background(Color.gray.opacity(0.2))
                    .cornerRadius(8)
                }
                .padding()
                .background(Color.white)
                .cornerRadius(16)
                .shadow(radius: 10)
                .padding(.horizontal, 20)
            }
        }
    }

    // MARK: - Helper Functions
    
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
            if granted {
                print("Microphone permission granted")
            } else {
                print("Microphone permission denied")
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
                        // Automatically send the API request after speech is complete
                        if !self.userInput.isEmpty {
                            self.sendToAPI()
                        }
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
        userInput = ""

        guard let url = URL(string: "https://f8cb321bfd70.ngrok-free.app/scheduler/process_input") else {
            messages.append(ChatMessage(text: "Invalid URL.", isUserMessage: false))
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let requestBody = ["input_text": currentInput]
        
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

        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: eventDetails)
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

        // Combine event details and calendar insights for the request body
        let requestBody: [String: Any] = [
            "event_details": eventDetails,
            "calendar_insights": calendarInsights
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
        HomePage()
    }
}
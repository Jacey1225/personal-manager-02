import SwiftUI

// MARK: - Discussion Models

struct DiscussionData: Codable {
    let title: String
    let author_id: String
    let active_contributors: [String]
    let content: [MessageContent] // Array of message objects
    let created_time: String
    let transparency: Bool
}

struct MessageContent: Codable {
    let username: String
    let message: String
    let timestamp: String
}

struct Discussion: Codable, Identifiable {
    let id = UUID()
    let discussion_id: String
    let project_id: String
    let data: DiscussionData
    
    // Computed properties for easier access
    var title: String { data.title }
    var author_id: String { data.author_id }
    var member_ids: [String] { data.active_contributors }
    var created_at: String { data.created_time }
    var messages: [DiscussionMessage] {
        return data.content.enumerated().map { index, messageContent in
            DiscussionMessage(
                message_id: "\(discussion_id)_\(index)",
                author_id: messageContent.username,
                content: messageContent.message,
                timestamp: messageContent.timestamp
            )
        }
    }
    
    private enum CodingKeys: String, CodingKey {
        case discussion_id
        case project_id
        case data
    }
}

struct DiscussionMessage: Codable, Identifiable {
    let id = UUID()
    let message_id: String
    let author_id: String
    let content: String
    let timestamp: String
    
    private enum CodingKeys: String, CodingKey {
        case message_id
        case author_id
        case content
        case timestamp
    }
}

// MARK: - Main Discussions View

struct DiscussionsView: View {
    let userId: String
    let projectId: String
    
    @State private var discussions: [Discussion] = []
    @State private var isLoadingDiscussions = false
    @State private var selectedDiscussion: Discussion? = nil
    @State private var showingCreateDiscussion = false
    
    var body: some View {
        VStack {
            if selectedDiscussion != nil {
                // Show discussion detail view
                DiscussionDetailView(
                    discussion: selectedDiscussion!,
                    userId: userId,
                    projectId: projectId,
                    onBack: {
                        selectedDiscussion = nil
                    },
                    onDiscussionUpdated: {
                        fetchDiscussions()
                    }
                )
            } else {
                // Show discussions list
                VStack(spacing: 12) {
                    // Header with create button
                    HStack {
                        Text("Discussions")
                            .font(.headline)
                            .fontWeight(.semibold)
                        
                        Spacer()
                        
                        Button(action: {
                            showingCreateDiscussion = true
                        }) {
                            Image(systemName: "plus.circle.fill")
                                .foregroundColor(.blue)
                                .font(.title2)
                        }
                    }
                    .padding(.horizontal)
                    
                    if isLoadingDiscussions {
                        ProgressView("Loading discussions...")
                            .frame(maxWidth: .infinity, minHeight: 150)
                    } else if discussions.isEmpty {
                        VStack(spacing: 16) {
                            Image(systemName: "bubble.left.and.bubble.right")
                                .font(.system(size: 40))
                                .foregroundColor(.gray)
                            
                            Text("No Discussions Yet")
                                .font(.headline)
                                .fontWeight(.semibold)
                            
                            Text("Start a discussion about this project")
                                .foregroundColor(.secondary)
                                .multilineTextAlignment(.center)
                                .font(.caption)
                            
                            Button("Create Discussion") {
                                showingCreateDiscussion = true
                            }
                            .buttonStyle(.borderedProminent)
                        }
                        .frame(maxWidth: .infinity, minHeight: 150)
                    } else {
                        ScrollView {
                            LazyVStack(spacing: 8) {
                                ForEach(discussions) { discussion in
                                    DiscussionRowView(
                                        discussion: discussion,
                                        userId: userId,
                                        onTap: {
                                            selectedDiscussion = discussion
                                        }
                                    )
                                    .padding(.horizontal)
                                }
                            }
                            .padding(.vertical)
                        }
                    }
                }
            }
        }
        .onAppear {
            fetchDiscussions()
        }
        .sheet(isPresented: $showingCreateDiscussion) {
            CreateDiscussionSheet(
                userId: userId,
                projectId: projectId,
                onDiscussionCreated: {
                    fetchDiscussions()
                }
            )
        }
    }
    
    private func fetchDiscussions() {
        isLoadingDiscussions = true
        
        guard let url = URL(string: "https://29098e308ec4.ngrok-free.app/discussions/list_project_discussions") else {
            print("Invalid URL for project discussions")
            isLoadingDiscussions = false
            return
        }
        
        let requestBody: [String: Any] = [
            "user_id": userId,
            "project_id": projectId
        ]
        
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        // For GET requests, we can't send a body, so we'll include as query parameters
        var urlComponents = URLComponents(url: url, resolvingAgainstBaseURL: false)
        urlComponents?.queryItems = [
            URLQueryItem(name: "user_id", value: userId),
            URLQueryItem(name: "project_id", value: projectId)
        ]
        
        guard let finalUrl = urlComponents?.url else {
            print("Failed to create URL for project discussions")
            isLoadingDiscussions = false
            return
        }
        
        request = URLRequest(url: finalUrl)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                isLoadingDiscussions = false
                
                if let error = error {
                    print("Network error: \(error.localizedDescription)")
                    return
                }
                
                guard let data = data else {
                    print("No data received for project discussions")
                    return
                }
                
                do {
                    print("Raw response data: \(String(data: data, encoding: .utf8) ?? "Unable to decode raw data")")
                    let response = try JSONDecoder().decode([String: [Discussion]].self, from: data)
                    discussions = response["discussions"] ?? []
                    print("Successfully decoded \(discussions.count) discussions")
                    for discussion in discussions {
                        print("Discussion: \(discussion.title) created at \(discussion.created_at)")
                    }
                } catch {
                    print("Failed to decode discussions: \(error.localizedDescription)")
                    print("Raw data: \(String(data: data, encoding: .utf8) ?? "Unable to decode")")
                    discussions = []
                }
            }
        }.resume()
    }
}

// MARK: - Discussion Components

struct DiscussionRowView: View {
    let discussion: Discussion
    let userId: String
    let onTap: () -> Void
    
    var body: some View {
        Button(action: onTap) {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text(discussion.title)
                        .font(.headline)
                        .foregroundColor(.primary)
                        .multilineTextAlignment(.leading)
                    
                    Spacer()
                    
                    if discussion.author_id == userId {
                        Image(systemName: "crown.fill")
                            .foregroundColor(.yellow)
                            .font(.caption)
                    }
                }
                
                HStack {
                    Text("By \(discussion.author_id)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Spacer()
                    
                    Text("\(discussion.member_ids.count) members")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                // Add creation time display
                HStack {
                    Image(systemName: "clock")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    
                    Text("Created \(formatCreationTime(discussion.created_at))")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    
                    Spacer()
                }
            }
            .padding()
            .background(Color(.systemGray6))
            .cornerRadius(12)
        }
        .buttonStyle(PlainButtonStyle())
    }
    
    private func formatCreationTime(_ timestamp: String) -> String {
        let formatter = ISO8601DateFormatter()
        guard let date = formatter.date(from: timestamp) else {
            // Fallback: try to parse just the date part if it's in format "YYYY-MM-DD"
            let dateComponents = timestamp.components(separatedBy: "T").first ?? timestamp
            return dateComponents
        }
        
        let now = Date()
        let calendar = Calendar.current
        
        if calendar.isDateInToday(date) {
            let timeFormatter = DateFormatter()
            timeFormatter.timeStyle = .short
            return "today at \(timeFormatter.string(from: date))"
        } else if calendar.isDateInYesterday(date) {
            let timeFormatter = DateFormatter()
            timeFormatter.timeStyle = .short
            return "yesterday at \(timeFormatter.string(from: date))"
        } else {
            let daysDiff = calendar.dateComponents([.day], from: date, to: now).day ?? 0
            if daysDiff < 7 {
                let dayFormatter = DateFormatter()
                dayFormatter.dateFormat = "EEEE"
                let timeFormatter = DateFormatter()
                timeFormatter.timeStyle = .short
                return "on \(dayFormatter.string(from: date)) at \(timeFormatter.string(from: date))"
            } else {
                let dateFormatter = DateFormatter()
                dateFormatter.dateStyle = .medium
                return "on \(dateFormatter.string(from: date))"
            }
        }
    }
}

struct CreateDiscussionSheet: View {
    let userId: String
    let projectId: String
    let onDiscussionCreated: () -> Void
    
    @Environment(\.presentationMode) var presentationMode
    @State private var title = ""
    @State private var description = ""
    @State private var isLoading = false
    @State private var errorMessage = ""
    @State private var showingError = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Discussion Title")
                        .font(.headline)
                    
                    TextField("Enter discussion title", text: $title)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                }
                
                VStack(alignment: .leading, spacing: 8) {
                    Text("Description (Optional)")
                        .font(.headline)
                    
                    TextField("Enter description", text: $description, axis: .vertical)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .lineLimit(3...6)
                }
                
                Spacer()
            }
            .padding()
            .navigationTitle("New Discussion")
            .navigationBarTitleDisplayMode(.inline)
            .navigationBarItems(
                leading: Button("Cancel") {
                    presentationMode.wrappedValue.dismiss()
                },
                trailing: Button("Create") {
                    createDiscussion()
                }
                .disabled(title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isLoading)
            )
        }
        .alert("Error", isPresented: $showingError) {
            Button("OK", role: .cancel) { }
        } message: {
            Text(errorMessage)
        }
    }
    
    private func createDiscussion() {
        guard !title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        
        isLoading = true
        errorMessage = ""
        
        guard let url = URL(string: "https://29098e308ec4.ngrok-free.app/discussions/create_discussion") else {
            errorMessage = "Invalid URL"
            showingError = true
            isLoading = false
            return
        }
        
        // Create the request body matching FastAPI's multiple body parameter format
        let requestBody: [String: Any] = [
            "request": [
                "user_id": userId,
                "project_id": projectId
            ],
            "discussion_data": [
                "title": title.trimmingCharacters(in: .whitespacesAndNewlines),
                "author_id": userId,
                "active_contributors": [userId], // Start with just the author
                "content": [], // Empty content array initially  
                "transparency": true
            ]
        ]
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            errorMessage = "Failed to encode request"
            showingError = true
            isLoading = false
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
                
                if let httpResponse = response as? HTTPURLResponse {
                    if httpResponse.statusCode == 200 {
                        onDiscussionCreated()
                        presentationMode.wrappedValue.dismiss()
                    } else {
                        errorMessage = "Failed to create discussion (Status: \(httpResponse.statusCode))"
                        showingError = true
                    }
                }
            }
        }.resume()
    }
}

struct DiscussionDetailView: View {
    let discussion: Discussion
    let userId: String
    let projectId: String
    let onBack: () -> Void
    let onDiscussionUpdated: () -> Void
    
    @State private var messages: [DiscussionMessage] = []
    @State private var isLoadingMessages = false
    @State private var newMessageText = ""
    @State private var isPostingMessage = false
    @State private var isMember = false
    @State private var isJoiningLeaving = false
    @State private var showingDeleteAlert = false
    @State private var isDeleting = false
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            VStack(spacing: 12) {
                HStack {
                    Button(action: onBack) {
                        Image(systemName: "chevron.left")
                            .font(.title2)
                            .foregroundColor(.blue)
                    }
                    
                    Spacer()
                    
                    if discussion.author_id == userId {
                        Button("Delete") {
                            showingDeleteAlert = true
                        }
                        .foregroundColor(.red)
                        .disabled(isDeleting)
                    }
                }
                
                VStack(alignment: .leading, spacing: 8) {
                    Text(discussion.title)
                        .font(.title2)
                        .fontWeight(.bold)
                        .multilineTextAlignment(.leading)
                    
                    HStack {
                        Text("By \(discussion.author_id)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        
                        Spacer()
                        
                        Text("\(discussion.member_ids.count) members")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                
                // Join/Leave button
                if !isMember && discussion.author_id != userId {
                    Button(action: joinDiscussion) {
                        if isJoiningLeaving {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                .scaleEffect(0.8)
                        } else {
                            Text("Join Discussion")
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isJoiningLeaving)
                } else if isMember && discussion.author_id != userId {
                    Button(action: leaveDiscussion) {
                        if isJoiningLeaving {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .red))
                                .scaleEffect(0.8)
                        } else {
                            Text("Leave Discussion")
                        }
                    }
                    .buttonStyle(.bordered)
                    .foregroundColor(.red)
                    .disabled(isJoiningLeaving)
                }
            }
            .padding()
            .background(Color(.systemGray6))
            
            // Messages
            if isMember || discussion.author_id == userId {
                VStack(spacing: 0) {
                    // Messages list
                    ScrollView {
                        if isLoadingMessages {
                            ProgressView("Loading messages...")
                                .frame(maxWidth: .infinity, minHeight: 200)
                        } else if messages.isEmpty {
                            VStack(spacing: 16) {
                                Image(systemName: "message")
                                    .font(.system(size: 40))
                                    .foregroundColor(.gray)
                                
                                Text("No Messages Yet")
                                    .font(.headline)
                                    .fontWeight(.semibold)
                                
                                Text("Be the first to start the conversation")
                                    .foregroundColor(.secondary)
                                    .multilineTextAlignment(.center)
                                    .font(.caption)
                            }
                            .frame(maxWidth: .infinity, minHeight: 200)
                        } else {
                            LazyVStack(spacing: 8) {
                                ForEach(messages) { message in
                                    MessageRowView(
                                        message: message,
                                        userId: userId,
                                        onDelete: {
                                            deleteMessage(message)
                                        }
                                    )
                                    .padding(.horizontal)
                                }
                            }
                            .padding(.vertical)
                        }
                    }
                    
                    // Message input
                    HStack(spacing: 12) {
                        TextField("Type a message...", text: $newMessageText, axis: .vertical)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                            .lineLimit(1...4)
                        
                        Button(action: postMessage) {
                            if isPostingMessage {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                    .scaleEffect(0.8)
                            } else {
                                Image(systemName: "paperplane.fill")
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(newMessageText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isPostingMessage)
                    }
                    .padding()
                    .background(Color(.systemBackground))
                }
            } else {
                VStack(spacing: 16) {
                    Image(systemName: "lock.fill")
                        .font(.system(size: 40))
                        .foregroundColor(.gray)
                    
                    Text("Join to View Messages")
                        .font(.headline)
                        .fontWeight(.semibold)
                    
                    Text("You need to join this discussion to view and post messages")
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .font(.caption)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .onAppear {
            checkMembershipAndFetchMessages()
        }
        .alert("Delete Discussion", isPresented: $showingDeleteAlert) {
            Button("Cancel", role: .cancel) { }
            Button("Delete", role: .destructive) {
                deleteDiscussion()
            }
        } message: {
            Text("Are you sure you want to delete this discussion? This action cannot be undone.")
        }
    }
    
    private func checkMembershipAndFetchMessages() {
        isMember = discussion.member_ids.contains(userId) || discussion.author_id == userId
        if isMember {
            fetchMessages()
        }
    }
    
    private func fetchMessages() {
        isLoadingMessages = true
        
        guard let url = URL(string: "https://29098e308ec4.ngrok-free.app/discussions/view_discussion") else {
            print("Invalid URL for discussion messages")
            isLoadingMessages = false
            return
        }
        
        var urlComponents = URLComponents(url: url, resolvingAgainstBaseURL: false)
        urlComponents?.queryItems = [
            URLQueryItem(name: "user_id", value: userId),
            URLQueryItem(name: "project_id", value: projectId),
            URLQueryItem(name: "discussion_id", value: discussion.discussion_id)
        ]
        
        guard let finalUrl = urlComponents?.url else {
            print("Failed to create URL for discussion messages")
            isLoadingMessages = false
            return
        }
        
        var request = URLRequest(url: finalUrl)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                isLoadingMessages = false
                
                if let error = error {
                    print("Network error: \(error.localizedDescription)")
                    return
                }
                
                guard let data = data else {
                    print("No data received for discussion messages")
                    return
                }
                
                do {
                    // The backend wraps the discussion in a "discussion" field
                    let response = try JSONDecoder().decode([String: Discussion].self, from: data)
                    guard let discussionResponse = response["discussion"] else {
                        print("No discussion found in response")
                        messages = []
                        return
                    }
                    
                    messages = discussionResponse.messages
                    print("Successfully loaded \(messages.count) messages")
                } catch {
                    print("Failed to decode discussion messages: \(error.localizedDescription)")
                    messages = []
                }
            }
        }.resume()
    }
    
    private func joinDiscussion() {
        isJoiningLeaving = true
        
        guard let url = URL(string: "https://29098e308ec4.ngrok-free.app/discussions/add_member") else {
            print("Invalid URL for joining discussion")
            isJoiningLeaving = false
            return
        }
        
        var urlComponents = URLComponents(url: url, resolvingAgainstBaseURL: false)
        urlComponents?.queryItems = [
            URLQueryItem(name: "discussion_id", value: discussion.discussion_id)
        ]
        
        guard let finalUrl = urlComponents?.url else {
            print("Failed to create URL for joining discussion")
            isJoiningLeaving = false
            return
        }
        
        let requestBody: [String: Any] = [
            "user_id": userId,
            "project_id": projectId
        ]
        
        var request = URLRequest(url: finalUrl)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            print("Failed to encode request body")
            isJoiningLeaving = false
            return
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                isJoiningLeaving = false
                
                if let error = error {
                    print("Failed to join discussion: \(error.localizedDescription)")
                    return
                }
                
                if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 {
                    isMember = true
                    fetchMessages()
                    onDiscussionUpdated()
                }
            }
        }.resume()
    }
    
    private func leaveDiscussion() {
        isJoiningLeaving = true
        
        guard let url = URL(string: "https://29098e308ec4.ngrok-free.app/discussions/remove_member") else {
            print("Invalid URL for leaving discussion")
            isJoiningLeaving = false
            return
        }
        
        var urlComponents = URLComponents(url: url, resolvingAgainstBaseURL: false)
        urlComponents?.queryItems = [
            URLQueryItem(name: "discussion_id", value: discussion.discussion_id)
        ]
        
        guard let finalUrl = urlComponents?.url else {
            print("Failed to create URL for leaving discussion")
            isJoiningLeaving = false
            return
        }
        
        let requestBody: [String: Any] = [
            "user_id": userId,
            "project_id": projectId
        ]
        
        var request = URLRequest(url: finalUrl)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            print("Failed to encode request body")
            isJoiningLeaving = false
            return
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                isJoiningLeaving = false
                
                if let error = error {
                    print("Failed to leave discussion: \(error.localizedDescription)")
                    return
                }
                
                if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 {
                    isMember = false
                    messages = []
                    onDiscussionUpdated()
                }
            }
        }.resume()
    }
    
    private func postMessage() {
        guard !newMessageText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        
        isPostingMessage = true
        let messageContent = newMessageText.trimmingCharacters(in: .whitespacesAndNewlines)
        
        guard let url = URL(string: "https://29098e308ec4.ngrok-free.app/discussions/post_message") else {
            print("Invalid URL for posting message")
            isPostingMessage = false
            return
        }
        
        var urlComponents = URLComponents(url: url, resolvingAgainstBaseURL: false)
        urlComponents?.queryItems = [
            URLQueryItem(name: "discussion_id", value: discussion.discussion_id),
            URLQueryItem(name: "message", value: messageContent)
        ]
        
        guard let finalUrl = urlComponents?.url else {
            print("Failed to create URL for posting message")
            isPostingMessage = false
            return
        }
        
        let requestBody: [String: Any] = [
            "user_id": userId,
            "project_id": projectId
        ]
        
        var request = URLRequest(url: finalUrl)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            print("Failed to encode request body")
            isPostingMessage = false
            return
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                isPostingMessage = false
                
                if let error = error {
                    print("Failed to post message: \(error.localizedDescription)")
                    return
                }
                
                if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 {
                    newMessageText = ""
                    fetchMessages()
                }
            }
        }.resume()
    }
    
    private func deleteMessage(_ message: DiscussionMessage) {
        guard let url = URL(string: "https://29098e308ec4.ngrok-free.app/discussions/remove_message") else {
            print("Invalid URL for deleting message")
            return
        }
        
        var urlComponents = URLComponents(url: url, resolvingAgainstBaseURL: false)
        urlComponents?.queryItems = [
            URLQueryItem(name: "discussion_id", value: discussion.discussion_id),
            URLQueryItem(name: "message", value: message.message_id)
        ]
        
        guard let finalUrl = urlComponents?.url else {
            print("Failed to create URL for deleting message")
            return
        }
        
        let requestBody: [String: Any] = [
            "user_id": userId,
            "project_id": projectId
        ]
        
        var request = URLRequest(url: finalUrl)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            print("Failed to encode request body")
            return
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    print("Failed to delete message: \(error.localizedDescription)")
                    return
                }
                
                if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 {
                    fetchMessages()
                }
            }
        }.resume()
    }
    
    private func deleteDiscussion() {
        isDeleting = true
        
        guard let url = URL(string: "https://29098e308ec4.ngrok-free.app/discussions/delete_discussion") else {
            print("Invalid URL for deleting discussion")
            isDeleting = false
            return
        }
        
        var urlComponents = URLComponents(url: url, resolvingAgainstBaseURL: false)
        urlComponents?.queryItems = [
            URLQueryItem(name: "discussion_id", value: discussion.discussion_id)
        ]
        
        guard let finalUrl = urlComponents?.url else {
            print("Failed to create URL for deleting discussion")
            isDeleting = false
            return
        }
        
        let requestBody: [String: Any] = [
            "user_id": userId,
            "project_id": projectId
        ]
        
        var request = URLRequest(url: finalUrl)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            print("Failed to encode request body")
            isDeleting = false
            return
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                isDeleting = false
                
                if let error = error {
                    print("Failed to delete discussion: \(error.localizedDescription)")
                    return
                }
                
                if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 {
                    onDiscussionUpdated()
                    onBack()
                }
            }
        }.resume()
    }
}

struct MessageRowView: View {
    let message: DiscussionMessage
    let userId: String
    let onDelete: () -> Void
    
    var isCurrentUser: Bool {
        message.author_id == userId
    }
    
    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            if isCurrentUser {
                Spacer(minLength: 50) // Push current user messages to the right
            }
            
            VStack(alignment: isCurrentUser ? .trailing : .leading, spacing: 4) {
                // Author name and timestamp
                HStack {
                    if !isCurrentUser {
                        Text(message.author_id)
                            .font(.caption2)
                            .fontWeight(.medium)
                            .foregroundColor(.secondary)
                        
                        Spacer()
                    }
                    
                    Text(formatTimestamp(message.timestamp))
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    
                    if isCurrentUser {
                        Button(action: onDelete) {
                            Image(systemName: "trash")
                                .font(.caption2)
                                .foregroundColor(.red)
                        }
                        .padding(.leading, 4)
                    }
                }
                
                // Message bubble
                HStack {
                    if isCurrentUser {
                        Spacer()
                    }
                    
                    Text(message.content)
                        .font(.body)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(
                            RoundedRectangle(cornerRadius: 16, style: .continuous)
                                .fill(isCurrentUser ? Color.blue : Color(.systemGray5))
                        )
                        .foregroundColor(isCurrentUser ? .white : .primary)
                        .multilineTextAlignment(isCurrentUser ? .trailing : .leading)
                    
                    if !isCurrentUser {
                        Spacer()
                    }
                }
            }
            
            if !isCurrentUser {
                Spacer(minLength: 50) // Push other users' messages to the left
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 2)
    }
    
    private func formatTimestamp(_ timestamp: String) -> String {
        let formatter = ISO8601DateFormatter()
        guard let date = formatter.date(from: timestamp) else {
            // Fallback: try to parse just the date part if it's in format "YYYY-MM-DD"
            return timestamp.components(separatedBy: "T").first ?? timestamp
        }
        
        let timeFormatter = DateFormatter()
        timeFormatter.timeStyle = .short
        return timeFormatter.string(from: date)
    }
}

import SwiftUI

// MARK: - Data Models

struct Project: Codable, Identifiable {
    let id = UUID()
    let project_name: String
    let project_id: String
    let project_members: [(String, String)] // Changed from project_emails to project_members as tuples
    
    private enum CodingKeys: String, CodingKey {
        case project_name
        case project_id
        case project_members
    }
    
    // Custom Decodable implementation since tuples aren't Codable
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        project_name = try container.decode(String.self, forKey: .project_name)
        project_id = try container.decode(String.self, forKey: .project_id)
        
        // Decode as array of arrays and convert to tuples
        let membersArray = try container.decode([[String]].self, forKey: .project_members)
        project_members = membersArray.compactMap { array in
            guard array.count >= 2 else { return nil }
            return (array[0], array[1])
        }
    }
    
    // Custom Encodable implementation
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(project_name, forKey: .project_name)
        try container.encode(project_id, forKey: .project_id)
        
        // Convert tuples to array of arrays for encoding
        let membersArray = project_members.map { [$0.0, $0.1] }
        try container.encode(membersArray, forKey: .project_members)
    }
    
    // Helper computed property to get just the emails for compatibility
    var memberEmails: [String] {
        return project_members.map { $0.0 }
    }
    
    // Helper computed property to get formatted member display
    var memberDisplayText: [String] {
        return project_members.map { "\($0.1) (\($0.0))" }
    }
}

struct ProjectEvent: Codable, Identifiable {
    let id = UUID()
    let event_name: String
    let start: String
    let end: String
    let description: String?
    let is_event: Bool?
    let event_id: String?
    
    private enum CodingKeys: String, CodingKey {
        case event_name
        case start
        case end
        case description
        case is_event
        case event_id
    }
    
    var formattedStartTime: String {
        return formatDateTime(start)
    }
    
    var formattedEndTime: String {
        return formatDateTime(end)
    }
    
    private func formatDateTime(_ dateString: String) -> String {
        // Handle both ISO format and human-readable format
        if dateString.contains("T") {
            // ISO format - convert to readable format
            let formatter = ISO8601DateFormatter()
            formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            
            if let date = formatter.date(from: dateString) {
                let displayFormatter = DateFormatter()
                displayFormatter.dateStyle = .medium
                displayFormatter.timeStyle = .short
                return displayFormatter.string(from: date)
            }
        }
        // Already in human-readable format, return as-is
        return dateString
    }
}

struct UserAvailability: Codable {
    let username: String
    let isAvailable: Bool
}

// MARK: - Main Projects View

struct ProjectsView: View {
    let userId: String
    
    @State private var projects: [Project] = []
    @State private var isLoading = false
    @State private var errorMessage = ""
    @State private var showingError = false
    
    // Create Project Sheet
    @State private var showingCreateProject = false
    @State private var newProjectName = ""
    @State private var newProjectEmails = ""
    
    // Selected Project State
    @State private var selectedProject: Project? = nil
    
    var body: some View {
        NavigationView {
            VStack {
                if isLoading {
                    ProgressView("Loading projects...")
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if projects.isEmpty {
                    emptyStateView
                } else {
                    projectsList
                }
            }
            .navigationTitle("Projects")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("New Project") {
                        showingCreateProject = true
                    }
                    .foregroundColor(.blue)
                }
            }
            .onAppear {
                fetchProjects()
            }
            .alert("Error", isPresented: $showingError) {
                Button("OK") { }
            } message: {
                Text(errorMessage)
            }
            .sheet(isPresented: $showingCreateProject) {
                CreateProjectSheet(
                    userId: userId,
                    onProjectCreated: {
                        fetchProjects()
                    }
                )
            }
            .sheet(item: $selectedProject) { project in
                ProjectDetailView(
                    userId: userId,
                    project: project
                )
            }
        }
    }
    
    private var emptyStateView: some View {
        VStack(spacing: 20) {
            Image(systemName: "folder.badge.plus")
                .font(.system(size: 60))
                .foregroundColor(.gray)
            
            Text("No Projects Yet")
                .font(.title2)
                .fontWeight(.semibold)
            
            Text("Create your first project to start collaborating")
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
            
            Button("Create Project") {
                showingCreateProject = true
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
    }
    
    private var projectsList: some View {
        List(projects) { project in
            ProjectRowView(project: project) {
                selectedProject = project
            }
        }
        .refreshable {
            fetchProjects()
        }
    }
    
    // MARK: - API Functions
    
    private func fetchProjects() {
        isLoading = true
        errorMessage = ""
        
    guard let url = URL(string: "https://29098e308ec4.ngrok-free.app/projects/list") else {
            errorMessage = "Invalid URL"
            showingError = true
            isLoading = false
            return
        }
        
        var urlComponents = URLComponents(url: url, resolvingAgainstBaseURL: false)
        urlComponents?.queryItems = [
            URLQueryItem(name: "user_id", value: userId)
        ]
        
        guard let finalUrl = urlComponents?.url else {
            errorMessage = "Failed to create URL"
            showingError = true
            isLoading = false
            return
        }
        
        var request = URLRequest(url: finalUrl)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
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
                
                do {
                    let projectsResponse = try JSONDecoder().decode([Project].self, from: data)
                    projects = projectsResponse
                } catch {
                    errorMessage = "Failed to decode projects: \(error.localizedDescription)"
                    showingError = true
                }
            }
        }.resume()
    }
}

// MARK: - Project Row View

struct ProjectRowView: View {
    let project: Project
    let onTap: () -> Void
    
    var body: some View {
        Button(action: onTap) {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Image(systemName: "folder.fill")
                        .foregroundColor(.blue)
                        .font(.title2)
                    
                    VStack(alignment: .leading) {
                        Text(project.project_name)
                            .font(.headline)
                            .foregroundColor(.primary)
                        
                        Text("\(project.project_members.count) member\(project.project_members.count == 1 ? "" : "s")")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    
                    Spacer()
                    
                    Image(systemName: "chevron.right")
                        .foregroundColor(.secondary)
                        .font(.caption)
                }
                
                // Show first few members with usernames
                if !project.project_members.isEmpty {
                    let displayMembers = project.memberDisplayText.prefix(2)
                    let remainingCount = project.project_members.count - displayMembers.count
                    
                    VStack(alignment: .leading, spacing: 2) {
                        ForEach(Array(displayMembers), id: \.self) { memberText in
                            Text(memberText)
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                        
                        if remainingCount > 0 {
                            Text("and \(remainingCount) more...")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                                .italic()
                        }
                    }
                }
            }
            .padding(.vertical, 4)
        }
        .buttonStyle(PlainButtonStyle())
    }
}

// MARK: - Create Project Sheet

struct CreateProjectSheet: View {
    let userId: String
    let onProjectCreated: () -> Void
    
    @Environment(\.presentationMode) var presentationMode
    @State private var projectName = ""
    @State private var membersText = ""
    @State private var isCreating = false
    @State private var errorMessage = ""
    @State private var showingError = false
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Project Details")) {
                    TextField("Project Name", text: $projectName)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                }
                
                Section(
                    header: Text("Project Members"), 
                    footer: Text("Enter members as 'email,username' pairs, one per line.\nExample:\njohn@example.com,John Doe\njane@example.com,Jane Smith")
                ) {
                    TextEditor(text: $membersText)
                        .frame(minHeight: 120)
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(Color.gray.opacity(0.3), lineWidth: 1)
                        )
                }
            }
            .navigationTitle("New Project")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        presentationMode.wrappedValue.dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Create") {
                        createProject()
                    }
                    .disabled(projectName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isCreating)
                }
            }
            .alert("Error", isPresented: $showingError) {
                Button("OK") { }
            } message: {
                Text(errorMessage)
            }
        }
    }
    
    private func createProject() {
        isCreating = true
        errorMessage = ""
        
        let trimmedName = projectName.trimmingCharacters(in: .whitespacesAndNewlines)
        
        // Parse members from text input
        let memberTuples = parseMembers(from: membersText)
        
        if memberTuples.isEmpty {
            errorMessage = "Please add at least one member"
            showingError = true
            isCreating = false
            return
        }
        
        guard let url = URL(string: "https://29098e308ec4.ngrok-free.app/projects/create_project") else {
            errorMessage = "Invalid URL"
            showingError = true
            isCreating = false
            return
        }
        
        let requestBody: [String: Any] = [
            "project_name": trimmedName,
            "project_members": memberTuples.map { [$0.0, $0.1] }, // Convert tuples to arrays for JSON
            "user_id": userId
        ]
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            errorMessage = "Failed to encode project data"
            showingError = true
            isCreating = false
            return
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                isCreating = false
                
                if let error = error {
                    errorMessage = "Network error: \(error.localizedDescription)"
                    showingError = true
                    return
                }
                
                // Success - close sheet and refresh projects
                presentationMode.wrappedValue.dismiss()
                onProjectCreated()
            }
        }.resume()
    }
    
    private func parseMembers(from text: String) -> [(String, String)] {
        let lines = text.split(separator: "\n")
        var members: [(String, String)] = []
        
        for line in lines {
            let trimmedLine = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmedLine.isEmpty {
                let components = trimmedLine.split(separator: ",", maxSplits: 1)
                if components.count == 2 {
                    let email = String(components[0]).trimmingCharacters(in: .whitespaces)
                    let username = String(components[1]).trimmingCharacters(in: .whitespaces)
                    members.append((email, username))
                }
            }
        }
        
        return members
    }
}

// MARK: - Add Member Sheet

struct AddMemberSheet: View {
    let userId: String
    let projectId: String
    let onMemberAdded: () -> Void
    
    @Environment(\.presentationMode) var presentationMode
    @State private var newMemberEmail = ""
    @State private var newMemberUsername = ""
    @State private var isAdding = false
    @State private var errorMessage = ""
    @State private var showingError = false
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Member Details")) {
                    TextField("Email address", text: $newMemberEmail)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .keyboardType(.emailAddress)
                        .autocapitalization(.none)
                    
                    TextField("Username", text: $newMemberUsername)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .autocapitalization(.words)
                }
            }
            .navigationTitle("Add Member")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        presentationMode.wrappedValue.dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Add") {
                        addMember()
                    }
                    .disabled(newMemberEmail.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || 
                             newMemberUsername.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || 
                             isAdding)
                }
            }
            .alert("Error", isPresented: $showingError) {
                Button("OK") { }
            } message: {
                Text(errorMessage)
            }
        }
    }
    
    private func addMember() {
        isAdding = true
        errorMessage = ""
        
        let trimmedEmail = newMemberEmail.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedUsername = newMemberUsername.trimmingCharacters(in: .whitespacesAndNewlines)
        
        guard let url = URL(string: "https://29098e308ec4.ngrok-free.app/projects/add_member") else {
            errorMessage = "Invalid URL"
            showingError = true
            isAdding = false
            return
        }
        
        var urlComponents = URLComponents(url: url, resolvingAgainstBaseURL: false)
        urlComponents?.queryItems = [
            URLQueryItem(name: "project_id", value: projectId),
            URLQueryItem(name: "user_id", value: userId),
            URLQueryItem(name: "new_email", value: trimmedEmail),
            URLQueryItem(name: "new_username", value: trimmedUsername)
        ]
        
        guard let finalUrl = urlComponents?.url else {
            errorMessage = "Failed to create URL"
            showingError = true
            isAdding = false
            return
        }
        
        var request = URLRequest(url: finalUrl)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                isAdding = false
                
                if let error = error {
                    errorMessage = "Network error: \(error.localizedDescription)"
                    showingError = true
                    return
                }
                
                // Success - close sheet
                presentationMode.wrappedValue.dismiss()
                onMemberAdded()
            }
        }.resume()
    }
}

// MARK: - Project Detail View

struct ProjectDetailView: View {
    let userId: String
    let project: Project
    
    @Environment(\.presentationMode) var presentationMode
    @State private var events: [ProjectEvent] = []
    @State private var isLoadingEvents = false
    @State private var errorMessage = ""
    @State private var showingError = false
    
    // Add Member Sheet
    @State private var showingAddMember = false
    
    // Availability Check Sheet
    @State private var showingAvailabilityCheck = false
    @State private var availabilityResults: [UserAvailability] = []
    @State private var availabilityPercentage: Double = 0.0
    @State private var showingAvailabilityResults = false
    
    var body: some View {
        NavigationView {
            VStack {
                // Action Buttons
                actionButtonsSection
                
                Divider()
                
                // Events List
                if isLoadingEvents {
                    ProgressView("Loading events...")
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if events.isEmpty {
                    emptyEventsView
                } else {
                    eventsList
                }
            }
            .navigationTitle(project.project_name)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        presentationMode.wrappedValue.dismiss()
                    }
                }
            }
            .onAppear {
                fetchProjectEvents()
            }
            .alert("Error", isPresented: $showingError) {
                Button("OK") { }
            } message: {
                Text(errorMessage)
            }
            .sheet(isPresented: $showingAddMember) {
                AddMemberSheet(
                    userId: userId,
                    projectId: project.project_id,
                    onMemberAdded: {
                        // Refresh project data would require parent callback
                    }
                )
            }
            .sheet(isPresented: $showingAvailabilityCheck) {
                AvailabilityCheckSheet(
                    userId: userId,
                    projectMembers: project.project_members,
                    onAvailabilityChecked: { results, percentage in
                        availabilityResults = results
                        availabilityPercentage = percentage
                        showingAvailabilityResults = true
                    }
                )
            }
            .sheet(isPresented: $showingAvailabilityResults) {
                AvailabilityResultsSheet(
                    availability: availabilityResults,
                    percentage: availabilityPercentage
                )
            }
        }
    }
    
    private var actionButtonsSection: some View {
        VStack(spacing: 12) {
            // Project Info with Member Management
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Members:")
                        .font(.headline)
                    
                    Spacer()
                    
                    Button("Add Member") {
                        showingAddMember = true
                    }
                    .font(.caption)
                    .buttonStyle(.borderedProminent)
                    .controlSize(.mini)
                }
                
                ForEach(Array(project.project_members.enumerated()), id: \.element.0) { index, member in
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(member.1) // Username
                                .font(.caption)
                                .fontWeight(.medium)
                            Text(member.0) // Email
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                        
                        Spacer()
                        
                        Button(action: {
                            deleteMember(email: member.0, username: member.1)
                        }) {
                            Image(systemName: "trash")
                                .foregroundColor(.red)
                                .font(.caption)
                        }
                        .buttonStyle(.borderless)
                    }
                    .padding(.vertical, 2)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding()
            .background(Color.gray.opacity(0.1))
            .cornerRadius(8)
            
            // Availability Check Button
            Button("Check Availability") {
                showingAvailabilityCheck = true
            }
            .buttonStyle(.bordered)
            .frame(maxWidth: .infinity)
        }
        .padding()
    }
    
    private var emptyEventsView: some View {
        VStack(spacing: 20) {
            Image(systemName: "calendar.badge.plus")
                .font(.system(size: 50))
                .foregroundColor(.gray)
            
            Text("No Project Events")
                .font(.title3)
                .fontWeight(.semibold)
            
            Text("Events related to this project will appear here")
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding()
    }
    
    private var eventsList: some View {
        List(events) { event in
            ProjectEventRowView(event: event)
        }
        .refreshable {
            fetchProjectEvents()
        }
    }
    
    // MARK: - API Functions
    
    private func fetchProjectEvents() {
        isLoadingEvents = true
        errorMessage = ""
        
        guard let url = URL(string: "https://29098e308ec4.ngrok-free.app/projects/events/\(project.project_id)") else {
            errorMessage = "Invalid URL"
            showingError = true
            isLoadingEvents = false
            return
        }
        
        var urlComponents = URLComponents(url: url, resolvingAgainstBaseURL: false)
        urlComponents?.queryItems = [
            URLQueryItem(name: "user_id", value: userId)
        ]
        
        guard let finalUrl = urlComponents?.url else {
            errorMessage = "Failed to create URL"
            showingError = true
            isLoadingEvents = false
            return
        }
        
        var request = URLRequest(url: finalUrl)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                isLoadingEvents = false
                
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
                
                do {
                    let eventsResponse = try JSONDecoder().decode([ProjectEvent].self, from: data)
                    events = eventsResponse
                } catch {
                    errorMessage = "Failed to decode events: \(error.localizedDescription)"
                    showingError = true
                }
            }
        }.resume()
    }
    
    private func deleteMember(email: String, username: String) {
        guard let url = URL(string: "https://29098e308ec4.ngrok-free.app/projects/delete_member") else {
            errorMessage = "Invalid URL"
            showingError = true
            return
        }
        
        var urlComponents = URLComponents(url: url, resolvingAgainstBaseURL: false)
        urlComponents?.queryItems = [
            URLQueryItem(name: "project_id", value: project.project_id),
            URLQueryItem(name: "user_id", value: userId),
            URLQueryItem(name: "email", value: email),
            URLQueryItem(name: "username", value: username)
        ]
        
        guard let finalUrl = urlComponents?.url else {
            errorMessage = "Failed to create URL"
            showingError = true
            return
        }
        
        var request = URLRequest(url: finalUrl)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    errorMessage = "Network error: \(error.localizedDescription)"
                    showingError = true
                    return
                }
                
                // Success - member deleted
                // Note: In a real app, you'd want to refresh the project data
                // For now, we'll just show a success message
            }
        }.resume()
    }
}

// MARK: - Project Event Row View

struct ProjectEventRowView: View {
    let event: ProjectEvent
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(event.event_name)
                .font(.headline)
            
            HStack {
                Image(systemName: "clock")
                    .foregroundColor(.blue)
                    .font(.caption)
                
                Text(event.formattedStartTime)
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                if !event.formattedEndTime.isEmpty && event.formattedEndTime != event.formattedStartTime {
                    Text("to")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Text(event.formattedEndTime)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            
            // Description is now hidden from the UI
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Availability Check Sheet

struct AvailabilityCheckSheet: View {
    let userId: String
    let projectMembers: [(String, String)] // Changed from projectEmails to projectMembers
    let onAvailabilityChecked: ([UserAvailability], Double) -> Void
    
    @Environment(\.presentationMode) var presentationMode
    @State private var startDateTime = Date()
    @State private var endDateTime = Date().addingTimeInterval(3600)
    @State private var isChecking = false
    @State private var errorMessage = ""
    @State private var showingError = false
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Time Range")) {
                    DatePicker("Start Time", selection: $startDateTime, displayedComponents: [.date, .hourAndMinute])
                    DatePicker("End Time", selection: $endDateTime, displayedComponents: [.date, .hourAndMinute])
                }
                
                Section(header: Text("Members to Check")) {
                    ForEach(projectMembers, id: \.0) { member in
                        VStack(alignment: .leading, spacing: 2) {
                            Text(member.1) // Username
                                .font(.caption)
                                .fontWeight(.medium)
                            Text(member.0) // Email
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                    }
                }
            }
            .navigationTitle("Check Availability")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        presentationMode.wrappedValue.dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Check") {
                        checkAvailability()
                    }
                    .disabled(isChecking || startDateTime >= endDateTime)
                }
            }
            .alert("Error", isPresented: $showingError) {
                Button("OK") { }
            } message: {
                Text(errorMessage)
            }
        }
    }
    
    private func checkAvailability() {
        isChecking = true
        errorMessage = ""
        
        // First fetch users
        fetchUsers { users in
            guard !users.isEmpty else {
                DispatchQueue.main.async {
                    errorMessage = "No users found"
                    showingError = true
                    isChecking = false
                }
                return
            }
            
            // Then check availability
            getAvailability(users: users)
        }
    }
    
    private func fetchUsers(completion: @escaping ([[String: Any]]) -> Void) {
        guard let url = URL(string: "https://29098e308ec4.ngrok-free.app/coordinate/fetch_users") else {
            errorMessage = "Invalid URL"
            showingError = true
            isChecking = false
            return
        }
        
        // Convert project members (tuples) to array of dictionaries
        let membersArray = projectMembers.map { member in
            return ["email": member.0, "username": member.1]
        }
        
        let requestBody: [String: Any] = [
            "members": membersArray
        ]
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            errorMessage = "Failed to encode request body"
            showingError = true
            isChecking = false
            return
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                DispatchQueue.main.async {
                    errorMessage = "Network error: \(error.localizedDescription)"
                    showingError = true
                    isChecking = false
                }
                return
            }
            
            // Validate HTTP response
            if let httpResponse = response as? HTTPURLResponse {
                guard (200...299).contains(httpResponse.statusCode) else {
                    DispatchQueue.main.async {
                        errorMessage = "Server error: HTTP \(httpResponse.statusCode)"
                        showingError = true
                        isChecking = false
                    }
                    return
                }
            }
            
            guard let data = data else {
                DispatchQueue.main.async {
                    errorMessage = "No data received from fetch_users"
                    showingError = true
                    isChecking = false
                }
                return
            }
            
            do {
                if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let users = json["users"] as? [[String: Any]] {
                    // Validate that we have users data
                    guard !users.isEmpty else {
                        DispatchQueue.main.async {
                            errorMessage = "No users found for the provided members"
                            showingError = true
                            isChecking = false
                        }
                        return
                    }
                    
                    // Pass the users to the completion handler
                    completion(users)
                } else {
                    DispatchQueue.main.async {
                        errorMessage = "Invalid response format from fetch_users"
                        showingError = true
                        isChecking = false
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    errorMessage = "Failed to parse fetch_users response: \(error.localizedDescription)"
                    showingError = true
                    isChecking = false
                }
            }
        }.resume()
    }
    
    private func getAvailability(users: [[String: Any]]) {
        // Validate inputs first
        guard !users.isEmpty else {
            DispatchQueue.main.async {
                errorMessage = "No users provided for availability check"
                showingError = true
                isChecking = false
            }
            return
        }
        
        // Validate that all users have required fields
        for user in users {
            guard user["user_id"] != nil,
                  user["username"] != nil else {
                DispatchQueue.main.async {
                    errorMessage = "Invalid user data: missing required fields"
                    showingError = true
                    isChecking = false
                }
                return
            }
        }
        
        // Validate date range
        guard startDateTime < endDateTime else {
            DispatchQueue.main.async {
                errorMessage = "Start time must be before end time"
                showingError = true
                isChecking = false
            }
            return
        }
        
        guard let url = URL(string: "https://29098e308ec4.ngrok-free.app/coordinate/get_availability") else {
            DispatchQueue.main.async {
                errorMessage = "Invalid URL"
                showingError = true
                isChecking = false
            }
            return
        }
        
        let formatter = ISO8601DateFormatter()
        let startTimeString = formatter.string(from: startDateTime)
        let endTimeString = formatter.string(from: endDateTime)
        
        // Validate ISO format strings
        guard !startTimeString.isEmpty, !endTimeString.isEmpty else {
            DispatchQueue.main.async {
                errorMessage = "Failed to format date times"
                showingError = true
                isChecking = false
            }
            return
        }
        
        // Create the request body with individual parameters as expected by FastAPI
        let requestBody: [String: Any] = [
            "users": users,
            "request_start": startTimeString,
            "request_end": endTimeString
        ]
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            DispatchQueue.main.async {
                errorMessage = "Failed to encode request"
                showingError = true
                isChecking = false
            }
            return
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                isChecking = false
                
                if let error = error {
                    errorMessage = "Network error: \(error.localizedDescription)"
                    showingError = true
                    return
                }
                
                // Validate HTTP response
                if let httpResponse = response as? HTTPURLResponse {
                    guard (200...299).contains(httpResponse.statusCode) else {
                        errorMessage = "Server error: HTTP \(httpResponse.statusCode)"
                        showingError = true
                        return
                    }
                }
                
                guard let data = data else {
                    errorMessage = "No data received from server"
                    showingError = true
                    return
                }
                
                do {
                    if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                        // Validate response structure
                        guard let status = json["status"] as? String,
                              status == "success" else {
                            errorMessage = "Server returned error status"
                            showingError = true
                            return
                        }
                        
                        guard let usersArray = json["users"] as? [[Any]] else {
                            errorMessage = "Invalid response format: missing users array"
                            showingError = true
                            return
                        }
                        
                        guard let percentAvailable = json["percent_available"] as? Double else {
                            errorMessage = "Invalid response format: missing percent_available"
                            showingError = true
                            return
                        }
                        
                        let availability = usersArray.compactMap { userArray -> UserAvailability? in
                            guard userArray.count >= 2,
                                  let username = userArray[0] as? String,
                                  let isAvailable = userArray[1] as? Bool else {
                                return nil
                            }
                            return UserAvailability(username: username, isAvailable: isAvailable)
                        }
                        
                        // Validate that we got availability data
                        guard !availability.isEmpty else {
                            errorMessage = "No availability data received"
                            showingError = true
                            return
                        }
                        
                        presentationMode.wrappedValue.dismiss()
                        onAvailabilityChecked(availability, percentAvailable)
                    } else {
                        errorMessage = "Invalid response format: not a JSON object"
                        showingError = true
                    }
                } catch {
                    errorMessage = "Failed to parse availability response: \(error.localizedDescription)"
                    showingError = true
                }
            }
        }.resume()
    }
}

// MARK: - Availability Results Sheet

struct AvailabilityResultsSheet: View {
    let availability: [UserAvailability]
    let percentage: Double
    
    @Environment(\.presentationMode) var presentationMode
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Percentage display at the top
                VStack(spacing: 8) {
                    Text("Availability Overview")
                        .font(.headline)
                        .fontWeight(.semibold)
                    
                    Text("\(Int(percentage.rounded()))% Available")
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(percentage >= 50 ? .green : .red)
                        .padding()
                        .background(
                            RoundedRectangle(cornerRadius: 12)
                                .fill(Color.gray.opacity(0.1))
                        )
                }
                .padding(.top)
                
                // User availability list
                List(availability, id: \.username) { user in
                    HStack {
                        Text(user.username)
                            .font(.headline)
                            .foregroundColor(user.isAvailable ? .white : .white)
                        
                        Spacer()
                        
                        HStack {
                            Image(systemName: user.isAvailable ? "checkmark.circle.fill" : "xmark.circle.fill")
                                .foregroundColor(.white)
                                .font(.title3)
                            
                            Text(user.isAvailable ? "Available" : "Busy")
                                .font(.subheadline)
                                .fontWeight(.medium)
                                .foregroundColor(.white)
                        }
                    }
                    .padding(.vertical, 12)
                    .padding(.horizontal, 16)
                    .background(
                        RoundedRectangle(cornerRadius: 10)
                            .fill(user.isAvailable ? Color.green : Color.red)
                    )
                    .listRowBackground(Color.clear)
                    .listRowSeparator(.hidden)
                }
                .listStyle(PlainListStyle())
            }
            .navigationTitle("Availability Results")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        presentationMode.wrappedValue.dismiss()
                    }
                }
            }
        }
    }
}

struct ProjectsView_Previews: PreviewProvider {
    static var previews: some View {
        ProjectsView(userId: "preview-user-id")
    }
}

import SwiftUI

// MARK: - Data Models

struct Project: Codable, Identifiable, Equatable {
    let id = UUID()
    var project_name: String
    let project_id: String
    let project_members: [(String, String)] // Changed from project_emails to project_members as tuples
    var project_likes: Int?
    let project_transparency: Bool?
    var is_liked: Bool?
    
    // Equatable conformance
    static func == (lhs: Project, rhs: Project) -> Bool {
        return lhs.project_id == rhs.project_id &&
               lhs.project_name == rhs.project_name &&
               lhs.project_likes == rhs.project_likes &&
               lhs.is_liked == rhs.is_liked &&
               lhs.project_transparency == rhs.project_transparency &&
               lhs.project_members.count == rhs.project_members.count &&
               zip(lhs.project_members, rhs.project_members).allSatisfy { $0.0 == $1.0 && $0.1 == $1.1 }
    }
    
    private enum CodingKeys: String, CodingKey {
        case project_name
        case project_id
        case project_members
        case project_likes
        case project_transparency
        case is_liked
    }
    
    // Custom Decodable implementation since tuples aren't Codable
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        project_name = try container.decode(String.self, forKey: .project_name)
        project_id = try container.decode(String.self, forKey: .project_id)
        project_likes = try container.decodeIfPresent(Int.self, forKey: .project_likes)
        project_transparency = try container.decodeIfPresent(Bool.self, forKey: .project_transparency)
        is_liked = try container.decodeIfPresent(Bool.self, forKey: .is_liked)
        
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
        try container.encodeIfPresent(project_likes, forKey: .project_likes)
        try container.encodeIfPresent(project_transparency, forKey: .project_transparency)
        try container.encodeIfPresent(is_liked, forKey: .is_liked)
        
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

// MARK: - Project Tab Types

enum ProjectTab: String, CaseIterable {
    case events = "Events"
    case discussions = "Discussions"
    case resources = "Resources"
    case progress = "Progress"
    
    var icon: String {
        switch self {
        case .events: return "calendar"
        case .discussions: return "bubble.left.and.bubble.right"
        case .resources: return "folder"
        case .progress: return "chart.line.uptrend.xyaxis"
        }
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

// MARK: - API Response Models

struct ProjectViewResponse: Codable {
    let project: Project
    let user_data: UserData
}

struct UserData: Codable {
    let user_id: String
    let username: String?
    let email: String?
    let projects: [String: [String]]? // Changed to match backend structure: [project_id: [project_name, permission]]
    let projects_liked: [String]?
    let permission: String? // Current user's permission for a specific project (used in view_project response)
    
    // Helper computed properties
    var projectPermissions: [String: String] {
        guard let projects = projects else { return [:] }
        var permissions: [String: String] = [:]
        for (projectId, data) in projects {
            if data.count >= 2 {
                permissions[projectId] = data[1] // Permission is the second element
            }
        }
        return permissions
    }
    
    var projectNames: [String: String] {
        guard let projects = projects else { return [:] }
        var names: [String: String] = [:]
        for (projectId, data) in projects {
            if data.count >= 1 {
                names[projectId] = data[0] // Project name is the first element
            }
        }
        return names
    }
}

// MARK: - Removed Discussion Models
// Discussion models have been moved to DiscussionsView.swift

// MARK: - Main Projects View

struct ProjectsView: View {
    let userId: String
    
    @State private var projects: [Project] = []
    @State private var isLoading = false
    @State private var errorMessage = ""
    @State private var showingError = false
    
    // Main project display state
    @State private var selectedProject: Project? = nil
    @State private var selectedTab: ProjectTab = .events
    @State private var isLoadingProjectDetails = false
    @State private var projectDetails: Project? = nil
    @State private var userPermission: String = "view" // Add user permission state
    
    // Create Project Sheet
    @State private var showingCreateProject = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                if isLoading {
                    ProgressView("Loading projects...")
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if projects.isEmpty {
                    emptyStateView
                } else {
                    ScrollView {
                        VStack(spacing: 16) {
                            // Organizations widget
                            OrganizationsWidget(
                                userId: userId,
                                projects: projects,
                                onOrganizationChanged: {
                                    fetchProjects()
                                }
                            )
                            .padding(.horizontal)
                            
                            // Main project widget
                            if let mainProject = selectedProject ?? projects.first {
                                MainProjectWidget(
                                    project: projectDetails ?? mainProject,
                                    selectedTab: $selectedTab,
                                    userId: userId,
                                    userPermission: userPermission,
                                    isLoading: isLoadingProjectDetails,
                                    onProjectUpdated: {
                                        fetchProjects()
                                    }
                                )
                                .id("\((projectDetails ?? mainProject).project_id)-\((projectDetails ?? mainProject).is_liked ?? false)-\((projectDetails ?? mainProject).project_likes ?? 0)") // Force re-render when project or like status changes
                                .padding(.horizontal)
                            }
                            
                            // Other projects widget
                            if projects.count > 1 {
                                OtherProjectsWidget(
                                    projects: otherProjects,
                                    onProjectSelected: { project in
                                        selectProject(project)
                                    }
                                )
                                .padding(.horizontal)
                            }
                        }
                        .padding(.vertical)
                    }
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
    
    private var otherProjects: [Project] {
        if let selected = selectedProject {
            return projects.filter { $0.project_id != selected.project_id }
        } else {
            return Array(projects.dropFirst())
        }
    }
    
    private func selectProject(_ project: Project) {
        print("Selecting project: \(project.project_name) (ID: \(project.project_id))")
        
        // Move selected project to main display
        if let currentMain = selectedProject {
            // Add current main project back to the list if it's not already there
            if !projects.contains(where: { $0.project_id == currentMain.project_id }) {
                projects.append(currentMain)
            }
        }
        
        selectedProject = project
        selectedTab = .events // Reset to events tab
        
        // Fetch detailed project information
        fetchProjectDetails(for: project)
    }
    
    // MARK: - API Functions
    
    private func fetchProjects() {
        isLoading = true
        errorMessage = ""
        
        guard let url = URL(string: "http://192.168.1.188:8000/projects/list") else {
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
                    
                    // Auto-select first project and fetch its details
                    if let firstProject = projects.first {
                        selectedProject = firstProject
                        fetchProjectDetails(for: firstProject)
                    }
                } catch {
                    errorMessage = "Failed to decode projects: \(error.localizedDescription)"
                    showingError = true
                }
            }
        }.resume()
    }
    
    private func fetchProjectDetails(for project: Project) {
        isLoadingProjectDetails = true
        
        print("Fetching project details for: \(project.project_name) (ID: \(project.project_id))")

        guard let url = URL(string: "http://192.168.1.188:8000/projects/view_project") else {
            errorMessage = "Invalid URL"
            showingError = true
            isLoadingProjectDetails = false
            return
        }
        
        var urlComponents = URLComponents(url: url, resolvingAgainstBaseURL: false)
        urlComponents?.queryItems = [
            URLQueryItem(name: "project_id", value: project.project_id),
            URLQueryItem(name: "user_id", value: userId),
            URLQueryItem(name: "project_name", value: project.project_name)
        ]
        
        guard let finalUrl = urlComponents?.url else {
            errorMessage = "Failed to create URL"
            showingError = true
            isLoadingProjectDetails = false
            return
        }
        
        print("Calling view_project route: \(finalUrl.absoluteString)")
        
        var request = URLRequest(url: finalUrl)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                isLoadingProjectDetails = false
                
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
                    // Try to decode the new response structure with project and user data
                    let response = try JSONDecoder().decode(ProjectViewResponse.self, from: data)
                    var updatedProject = response.project
                    
                    // Check if the project is liked by comparing with user's projects_liked array
                    let projectsLiked = response.user_data.projects_liked ?? []
                    updatedProject.is_liked = projectsLiked.contains(project.project_id)
                    
                    // Store user permission for this project
                    // First try to get permission from the response, then fallback to user's projects data
                    if let responsePermission = response.user_data.permission {
                        userPermission = responsePermission
                    } else if let userProjects = response.user_data.projects,
                              let projectData = userProjects[project.project_id],
                              projectData.count >= 2 {
                        userPermission = projectData[1] // Permission is the second element
                    } else {
                        userPermission = "view" // Default fallback
                    }
                    
                    projectDetails = updatedProject
                    print("Successfully fetched project details for: \(updatedProject.project_name)")
                    print("Projects liked array: \(projectsLiked)")
                    print("Current project ID: \(project.project_id)")
                    print("Project is liked: \(updatedProject.is_liked ?? false)")
                    print("User permission: \(userPermission)")
                    print("Updated projectDetails.is_liked: \(projectDetails?.is_liked ?? false)")
                    
                } catch {
                    print("Failed to decode project response: \(error.localizedDescription)")
                    
                    // Fallback: Try to decode as just a Project (for backward compatibility)
                    do {
                        let detailedProject = try JSONDecoder().decode(Project.self, from: data)
                        projectDetails = detailedProject
                        print("Successfully fetched project details (fallback): \(detailedProject.project_name)")
                    } catch {
                        // If both decodings fail, use the basic project info
                        projectDetails = project
                        print("Failed to decode detailed project, using basic info: \(error.localizedDescription)")
                    }
                }
            }
        }.resume()
    }
}

// MARK: - Main Project Widget

struct MainProjectWidget: View {
    let project: Project
    @Binding var selectedTab: ProjectTab
    let userId: String
    let userPermission: String
    let isLoading: Bool
    let onProjectUpdated: () -> Void // Callback to refresh projects list
    
    @State private var events: [ProjectEvent] = []
    @State private var isLoadingEvents = false
    
    // Project management states
    @State private var showingDeleteAlert = false
    @State private var showingRenameSheet = false
    @State private var showingMemberManagement = false
    @State private var isLiking = false
    @State private var newProjectName = ""
    @State private var currentProject: Project
    
    init(project: Project, selectedTab: Binding<ProjectTab>, userId: String, userPermission: String, isLoading: Bool, onProjectUpdated: @escaping () -> Void) {
        self.project = project
        self._selectedTab = selectedTab
        self.userId = userId
        self.userPermission = userPermission
        self.isLoading = isLoading
        self.onProjectUpdated = onProjectUpdated
        self._currentProject = State(initialValue: project)
    }
    
    var body: some View {
        VStack(spacing: 0) {
            // Project header
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Image(systemName: "folder.fill")
                        .foregroundColor(.blue)
                        .font(.title2)
                    
                    VStack(alignment: .leading, spacing: 2) {
                        Text(currentProject.project_name)
                            .font(.title2)
                            .fontWeight(.bold)
                        
                        if let likes = currentProject.project_likes {
                            HStack {
                                Image(systemName: "heart.fill")
                                    .foregroundColor(.red)
                                    .font(.caption)
                                Text("\(likes) likes")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    
                    Spacer()
                    
                    // Project management buttons
                    HStack(spacing: 12) {
                        // Like/Unlike button - Available for all permission levels
                        Button(action: toggleLike) {
                            Image(systemName: (currentProject.is_liked ?? false) ? "heart.fill" : "heart")
                                .foregroundColor((currentProject.is_liked ?? false) ? .red : .gray)
                                .font(.title3)
                        }
                        .disabled(isLiking)
                        
                        // Member management button - Available for edit and admin
                        if userPermission == "edit" || userPermission == "admin" {
                            Button(action: {
                                showingMemberManagement = true
                            }) {
                                Image(systemName: "person.2")
                                    .foregroundColor(.purple)
                                    .font(.title3)
                            }
                        }
                        
                        // Rename button - Available for admin only
                        if userPermission == "admin" {
                            Button(action: {
                                showingRenameSheet = true
                            }) {
                                Image(systemName: "pencil")
                                    .foregroundColor(.blue)
                                    .font(.title3)
                            }
                        }
                        
                        // Delete button - Available for admin only
                        if userPermission == "admin" {
                            Button(action: {
                                showingDeleteAlert = true
                            }) {
                                Image(systemName: "trash")
                                    .foregroundColor(.red)
                                    .font(.title3)
                            }
                        }
                        
                        // Transparency indicator - Visible for all permission levels
                        if let transparency = currentProject.project_transparency {
                            Image(systemName: transparency ? "eye" : "eye.slash")
                                .foregroundColor(transparency ? .green : .orange)
                                .font(.caption)
                        }
                    }
                }
                
                // Members info
                Text("\(currentProject.project_members.count) member\(currentProject.project_members.count == 1 ? "" : "s")")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding()
            .background(Color.clear)
            .onChange(of: project) { newProject in
                // Update currentProject when the parent project changes
                print("MainProjectWidget: Project changed to \(newProject.project_name)")
                print("MainProjectWidget: New project is_liked: \(newProject.is_liked ?? false)")
                currentProject = newProject
                print("MainProjectWidget: Updated currentProject.is_liked: \(currentProject.is_liked ?? false)")
                // Also refetch events for the new project
                if selectedTab == .events {
                    fetchProjectEvents()
                }
            }
            
            // Tab slider
            HStack(spacing: 0) {
                ForEach(ProjectTab.allCases, id: \.self) { tab in
                    Button(action: {
                        selectedTab = tab
                        if tab == .events {
                            fetchProjectEvents()
                        }
                        // Note: DiscussionsView handles its own data loading
                    }) {
                        VStack(spacing: 4) {
                            Image(systemName: tab.icon)
                                .font(.title3)
                            Text(tab.rawValue)
                                .font(.caption)
                                .fontWeight(.medium)
                        }
                        .foregroundColor(selectedTab == tab ? .blue : .gray)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                    }
                    .buttonStyle(PlainButtonStyle())
                }
            }
            .background(Color.clear)
            
            // Content area
            VStack {
                if isLoading {
                    ProgressView("Loading project details...")
                        .frame(maxWidth: .infinity, minHeight: 200)
                } else {
                    switch selectedTab {
                    case .events:
                        eventsContent
                    case .discussions:
                        discussionsContent
                    case .resources:
                        comingSoonContent(for: "Resources")
                    case .progress:
                        comingSoonContent(for: "Progress")
                    }
                }
            }
            .frame(minHeight: 200)
        }
        .background(Color.clear)
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.white, lineWidth: 1)
        )
        .cornerRadius(12)
        .onAppear {
            print("MainProjectWidget onAppear: project.is_liked = \(project.is_liked ?? false)")
            print("MainProjectWidget onAppear: currentProject.is_liked = \(currentProject.is_liked ?? false)")
            
            // Ensure currentProject is synced with the project parameter
            if currentProject.project_id == project.project_id && 
               currentProject.is_liked != project.is_liked {
                currentProject = project
                print("MainProjectWidget onAppear: Updated currentProject.is_liked = \(currentProject.is_liked ?? false)")
            }
            
            if selectedTab == .events {
                fetchProjectEvents()
            }
            // Note: DiscussionsView handles its own data loading
        }
        .alert("Delete Project", isPresented: $showingDeleteAlert) {
            Button("Cancel", role: .cancel) { }
            Button("Delete", role: .destructive) {
                deleteProject()
            }
        } message: {
            Text("Are you sure you want to delete '\(currentProject.project_name)'? This action cannot be undone.")
        }
        .sheet(isPresented: $showingRenameSheet) {
            NavigationView {
                VStack(spacing: 20) {
                    Text("Rename Project")
                        .font(.title2)
                        .fontWeight(.bold)
                        .padding(.top)
                    
                    TextField("Project Name", text: $newProjectName)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .onAppear {
                            newProjectName = currentProject.project_name
                        }
                    
                    Spacer()
                }
                .padding()
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .navigationBarLeading) {
                        Button("Cancel") {
                            showingRenameSheet = false
                            newProjectName = ""
                        }
                    }
                    
                    ToolbarItem(placement: .navigationBarTrailing) {
                        Button("Save") {
                            renameProject()
                        }
                        .disabled(newProjectName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                }
            }
        }
        .sheet(isPresented: $showingMemberManagement) {
            ProjectMemberManagementSheet(
                projectId: currentProject.project_id,
                userId: userId,
                currentMembers: currentProject.project_members,
                onMembersUpdated: {
                    // Refresh project data when members change
                    onProjectUpdated()
                }
            )
        }
    }
    
    private var eventsContent: some View {
        VStack {
            if isLoadingEvents {
                ProgressView("Loading events...")
                    .frame(maxWidth: .infinity, minHeight: 150)
            } else if events.isEmpty {
                VStack(spacing: 16) {
                    Image(systemName: "calendar.badge.plus")
                        .font(.system(size: 40))
                        .foregroundColor(.gray)
                    
                    Text("No Project Events")
                        .font(.headline)
                        .fontWeight(.semibold)
                    
                    Text("Events related to this project will appear here")
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .font(.caption)
                }
                .frame(maxWidth: .infinity, minHeight: 150)
            } else {
                LazyVStack(spacing: 8) {
                    ForEach(events.prefix(3)) { event in
                        ProjectEventRowView(event: event)
                            .padding(.horizontal)
                    }
                    
                    if events.count > 3 {
                        Text("and \(events.count - 3) more events...")
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .padding(.horizontal)
                    }
                }
                .padding(.vertical)
            }
        }
    }
    
    private var discussionsContent: some View {
        DiscussionsView(
            userId: userId,
            projectId: currentProject.project_id
        )
    }
    
    private func comingSoonContent(for feature: String) -> some View {
        VStack(spacing: 16) {
            Image(systemName: "hammer.fill")
                .font(.system(size: 40))
                .foregroundColor(.orange)
            
            Text("\(feature) Coming Soon")
                .font(.headline)
                .fontWeight(.semibold)
            
            Text("This feature is currently in development")
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .font(.caption)
        }
        .frame(maxWidth: .infinity, minHeight: 150)
    }
    
    private func fetchProjectEvents() {
        isLoadingEvents = true

        guard let url = URL(string: "http://192.168.1.188:8000/projects/events/\(currentProject.project_id)") else {
            print("Invalid URL for project events")
            isLoadingEvents = false
            return
        }
        
        var urlComponents = URLComponents(url: url, resolvingAgainstBaseURL: false)
        urlComponents?.queryItems = [
            URLQueryItem(name: "user_id", value: userId)
        ]
        
        guard let finalUrl = urlComponents?.url else {
            print("Failed to create URL for project events")
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
                    print("Network error: \(error.localizedDescription)")
                    return
                }
                
                guard let data = data else {
                    print("No data received for project events")
                    return
                }
                
                do {
                    let eventsResponse = try JSONDecoder().decode([ProjectEvent].self, from: data)
                    events = eventsResponse
                } catch {
                    print("Failed to decode events: \(error.localizedDescription)")
                    events = []
                }
            }
        }.resume()
    }
    
    // MARK: - Project Management Actions
    
    private func toggleLike() {
        isLiking = true
        let isCurrentlyLiked = currentProject.is_liked ?? false
        let endpoint = isCurrentlyLiked ? "unlike_project" : "like_project"
        guard let url = URL(string: "http://192.168.1.188:8000/projects/\(endpoint)") else {
            isLiking = false
            return
        }
        
        let requestBody = [
            "project_id": currentProject.project_id,
            "user_id": userId,
            "project_name": currentProject.project_name
        ]
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            print("Failed to encode like request: \(error)")
            isLiking = false
            return
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                isLiking = false
                if let error = error {
                    print("Failed to \(endpoint): \(error.localizedDescription)")
                    return
                }
                
                // Update local state
                currentProject.is_liked = !isCurrentlyLiked
                if isCurrentlyLiked {
                    currentProject.project_likes = max(0, (currentProject.project_likes ?? 0) - 1)
                } else {
                    currentProject.project_likes = (currentProject.project_likes ?? 0) + 1
                }
                
                // Notify parent to refresh
                onProjectUpdated()
            }
        }.resume()
    }
    
    private func renameProject() {
        guard !newProjectName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        
        guard let url = URL(string: "http://192.168.1.188:8000/projects/rename_project") else { return }
        
        let requestBody = [
            "project_id": currentProject.project_id,
            "project_name": newProjectName.trimmingCharacters(in: .whitespacesAndNewlines),
            "user_id": userId
        ]
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            print("Failed to encode rename request: \(error)")
            return
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    print("Failed to rename project: \(error.localizedDescription)")
                    return
                }
                
                // Update local state
                currentProject.project_name = newProjectName.trimmingCharacters(in: .whitespacesAndNewlines)
                newProjectName = ""
                showingRenameSheet = false
                
                // Notify parent to refresh
                onProjectUpdated()
            }
        }.resume()
    }
    
    private func deleteProject() {
        guard let url = URL(string: "http://192.168.1.188:8000/projects/delete_project") else { return }

        let requestBody = [
            "project_id": currentProject.project_id,
            "user_id": userId,
            "project_name": currentProject.project_name
        ]
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            print("Failed to encode delete request: \(error)")
            return
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    print("Failed to delete project: \(error.localizedDescription)")
                    return
                }
                
                // Notify parent to refresh (project will be removed from list)
                onProjectUpdated()
            }
        }.resume()
    }
}

// MARK: - Other Projects Widget

struct OtherProjectsWidget: View {
    let projects: [Project]
    let onProjectSelected: (Project) -> Void
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Other Projects")
                    .font(.headline)
                    .fontWeight(.semibold)
                
                Spacer()
                
                Text("\(projects.count)")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.gray.opacity(0.2))
                    .cornerRadius(8)
            }
            
            LazyVStack(spacing: 8) {
                ForEach(projects) { project in
                    OtherProjectRowView(project: project) {
                        onProjectSelected(project)
                    }
                }
            }
        }
        .padding()
        .background(Color.clear)
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.white, lineWidth: 1)
        )
        .cornerRadius(12)
    }
}

// MARK: - Other Project Row View

struct OtherProjectRowView: View {
    let project: Project
    let onTap: () -> Void
    
    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 12) {
                Image(systemName: "folder")
                    .foregroundColor(.blue)
                    .font(.title3)
                
                VStack(alignment: .leading, spacing: 2) {
                    Text(project.project_name)
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(.primary)
                        .lineLimit(1)
                    
                    HStack {
                        Text("\(project.project_members.count) member\(project.project_members.count == 1 ? "" : "s")")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        
                        if let likes = project.project_likes {
                            Text("•")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                            
                            HStack(spacing: 2) {
                                Image(systemName: "heart.fill")
                                    .foregroundColor(.red)
                                    .font(.caption2)
                                Text("\(likes)")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                }
                
                Spacer()
                
                Image(systemName: "arrow.up.circle")
                    .foregroundColor(.blue)
                    .font(.caption)
            }
            .padding(.vertical, 8)
            .padding(.horizontal, 12)
            .background(Color.gray.opacity(0.05))
            .cornerRadius(8)
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
    @State private var isTransparent = true
    @State private var isCreating = false
    @State private var errorMessage = ""
    @State private var showingError = false
    @State private var members: [(String, String)] = [] // Array of (email, username) pairs
    @State private var showAddMembers = false
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Project Details")) {
                    TextField("Project Name", text: $projectName)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                    
                    Toggle("Public Project", isOn: $isTransparent)
                        .toggleStyle(SwitchToggleStyle())
                }
                
                Section(header: Text("Members (Optional)")) {
                    if !showAddMembers {
                        Button(action: {
                            showAddMembers = true
                            members.append(("", "")) // Add first empty member
                        }) {
                            HStack {
                                Image(systemName: "plus.circle")
                                Text("Add Members")
                            }
                            .foregroundColor(.blue)
                        }
                    } else {
                        ForEach(members.indices, id: \.self) { index in
                            VStack(spacing: 8) {
                                TextField("Email", text: Binding(
                                    get: { members[index].0 },
                                    set: { members[index].0 = $0 }
                                ))
                                .textFieldStyle(RoundedBorderTextFieldStyle())
                                .keyboardType(.emailAddress)
                                .autocapitalization(.none)
                                
                                TextField("Username", text: Binding(
                                    get: { members[index].1 },
                                    set: { members[index].1 = $0 }
                                ))
                                .textFieldStyle(RoundedBorderTextFieldStyle())
                                .autocapitalization(.none)
                            }
                            .swipeActions {
                                Button("Delete", role: .destructive) {
                                    members.remove(at: index)
                                    if members.isEmpty {
                                        showAddMembers = false
                                    }
                                }
                            }
                        }
                        
                        Button(action: {
                            members.append(("", ""))
                        }) {
                            HStack {
                                Image(systemName: "plus.circle")
                                Text("Add Another Member")
                            }
                            .foregroundColor(.blue)
                        }
                    }
                }
            }
            .navigationTitle("Create Project")
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
                    .disabled(isCreating || projectName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
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
        
        // Filter out empty members
        let validMembers = members.filter { !$0.0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !$0.1.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }

        guard let url = URL(string: "http://192.168.1.188:8000/projects/create_project") else {
            errorMessage = "Invalid URL"
            showingError = true
            isCreating = false
            return
        }
        
        let requestBody: [String: Any] = [
            "project_name": projectName.trimmingCharacters(in: .whitespacesAndNewlines),
            "project_transparency": isTransparent,
            "project_likes": 0,
            "project_members": validMembers.map { [$0.0, $0.1] }, // Convert tuples to arrays for JSON
            "user_id": userId
        ]
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            errorMessage = "Failed to encode request"
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
                
                if let httpResponse = response as? HTTPURLResponse {
                    guard (200...299).contains(httpResponse.statusCode) else {
                        errorMessage = "Server error: HTTP \(httpResponse.statusCode)"
                        showingError = true
                        return
                    }
                }
                
                // Close sheet and refresh projects
                presentationMode.wrappedValue.dismiss()
                onProjectCreated()
            }
        }.resume()
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

        guard let url = URL(string: "http://192.168.1.188:8000/projects/add_member") else {
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
    @State private var currentUserEmail: String = ""
    
    // Join/Leave Project Sheet
    @State private var showingJoinProject = false
    @State private var showingAddMember = false // Keep for other members
    
    // Availability Check Sheet
    @State private var showingAvailabilityCheck = false
    @State private var availabilityResults: [UserAvailability] = []
    @State private var availabilityPercentage: Double = 0.0
    @State private var showingAvailabilityResults = false
    
    // Computed property to check if current user is a member
    var isCurrentUserMember: Bool {
        return project.project_members.contains { member in
            member.0 == currentUserEmail // Check if user's email matches any member email
        }
    }
    
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
                fetchCurrentUserEmail()
            }
            .alert("Error", isPresented: $showingError) {
                Button("OK") { }
            } message: {
                Text(errorMessage)
            }
            .sheet(isPresented: $showingJoinProject) {
                JoinProjectSheet(
                    userId: userId,
                    projectId: project.project_id,
                    onProjectJoined: {
                        // Refresh project data would require parent callback
                        fetchCurrentUserEmail()
                    }
                )
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
                    
                    if isCurrentUserMember {
                        Button("Leave Project") {
                            leaveProject()
                        }
                        .font(.caption)
                        .buttonStyle(.bordered)
                        .controlSize(.mini)
                        .foregroundColor(.red)
                    } else {
                        Button("Join Project") {
                            showingJoinProject = true
                        }
                        .font(.caption)
                        .buttonStyle(.borderedProminent)
                        .controlSize(.mini)
                    }
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
                        
                        // Only show delete button for members if current user is also a member
                        if isCurrentUserMember && member.0 != currentUserEmail {
                            Button(action: {
                                deleteMember(email: member.0, username: member.1)
                            }) {
                                Image(systemName: "trash")
                                    .foregroundColor(.red)
                                    .font(.caption)
                            }
                            .buttonStyle(.borderless)
                        }
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
        
        guard let url = URL(string: "http://192.168.1.188:8000/projects/events/\(project.project_id)") else {
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
        guard let url = URL(string: "http://192.168.1.188:8000/projects/delete_member") else {
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
        request.httpMethod = "DELETE"
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
    
    private func fetchCurrentUserEmail() {
        // Fetch user data to get current user's email
        guard let url = URL(string: "http://192.168.1.188:8000/projects/view_project") else {
            print("Invalid URL for fetching user data")
            return
        }
        
        var urlComponents = URLComponents(url: url, resolvingAgainstBaseURL: false)
        urlComponents?.queryItems = [
            URLQueryItem(name: "project_id", value: project.project_id),
            URLQueryItem(name: "user_id", value: userId),
            URLQueryItem(name: "project_name", value: project.project_name)
        ]
        
        guard let finalUrl = urlComponents?.url else {
            print("Failed to create URL for fetching user data")
            return
        }
        
        var request = URLRequest(url: finalUrl)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    print("Network error fetching user data: \(error.localizedDescription)")
                    return
                }
                
                guard let data = data else {
                    print("No data received for user data")
                    return
                }
                
                do {
                    let response = try JSONDecoder().decode(ProjectViewResponse.self, from: data)
                    self.currentUserEmail = response.user_data.email ?? ""
                } catch {
                    print("Failed to decode user data: \(error.localizedDescription)")
                }
            }
        }.resume()
    }
    
    private func leaveProject() {
        guard !currentUserEmail.isEmpty else {
            errorMessage = "Unable to determine current user email"
            showingError = true
            return
        }
        
        // Find current user's username from project members
        guard let currentMember = project.project_members.first(where: { $0.0 == currentUserEmail }) else {
            errorMessage = "Current user not found in project members"
            showingError = true
            return
        }
        
        deleteMember(email: currentMember.0, username: currentMember.1)
    }
}

// MARK: - Join Project Sheet

struct JoinProjectSheet: View {
    let userId: String
    let projectId: String
    let onProjectJoined: () -> Void
    
    @Environment(\.presentationMode) var presentationMode
    @State private var projectCode = ""
    @State private var username = ""
    @State private var isJoining = false
    @State private var errorMessage = ""
    @State private var showingError = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Username")
                        .font(.headline)
                    
                    TextField("Enter your username", text: $username)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                }
                
                VStack(alignment: .leading, spacing: 8) {
                    Text("Project Code")
                        .font(.headline)
                    
                    TextField("Enter project code", text: $projectCode)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .textContentType(.password) // Hide the input for security
                }
                
                Text("You need the project code to join this project. Ask a project member for the code.")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.leading)
                
                Spacer()
            }
            .padding()
            .navigationTitle("Join Project")
            .navigationBarTitleDisplayMode(.inline)
            .navigationBarItems(
                leading: Button("Cancel") {
                    presentationMode.wrappedValue.dismiss()
                },
                trailing: Button("Join") {
                    joinProject()
                }
                .disabled(projectCode.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || 
                         username.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || 
                         isJoining)
            )
        }
        .alert("Error", isPresented: $showingError) {
            Button("OK", role: .cancel) { }
        } message: {
            Text(errorMessage)
        }
    }
    
    private func joinProject() {
        guard !projectCode.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              !username.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        
        isJoining = true
        errorMessage = ""

        guard let url = URL(string: "http://192.168.1.188:8000/projects/add_member") else {
            errorMessage = "Invalid URL"
            showingError = true
            isJoining = false
            return
        }
        
        var urlComponents = URLComponents(url: url, resolvingAgainstBaseURL: false)
        urlComponents?.queryItems = [
            URLQueryItem(name: "project_id", value: projectId),
            URLQueryItem(name: "user_id", value: userId),
            URLQueryItem(name: "new_email", value: ""), // Will need to get current user's email
            URLQueryItem(name: "new_username", value: username.trimmingCharacters(in: .whitespacesAndNewlines)),
            URLQueryItem(name: "code", value: projectCode.trimmingCharacters(in: .whitespacesAndNewlines))
        ]
        
        // We need to get the user's email first, so let's fetch it
        fetchUserEmailAndJoin()
    }
    
    private func fetchUserEmailAndJoin() {
        // First, get the user's email
        guard let userUrl = URL(string: "http://192.168.1.188:8000/projects/view_project") else {
            errorMessage = "Invalid URL for user data"
            showingError = true
            isJoining = false
            return
        }
        
        var urlComponents = URLComponents(url: userUrl, resolvingAgainstBaseURL: false)
        urlComponents?.queryItems = [
            URLQueryItem(name: "project_id", value: projectId),
            URLQueryItem(name: "user_id", value: userId),
            URLQueryItem(name: "project_name", value: "")
        ]
        
        guard let finalUrl = urlComponents?.url else {
            errorMessage = "Failed to create URL for user data"
            showingError = true
            isJoining = false
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
                    isJoining = false
                    return
                }
                
                guard let data = data else {
                    errorMessage = "No user data received"
                    showingError = true
                    isJoining = false
                    return
                }
                
                do {
                    let response = try JSONDecoder().decode(ProjectViewResponse.self, from: data)
                    let userEmail = response.user_data.email ?? ""
                    
                    // Now call the add member endpoint with the user's email
                    self.addMemberWithEmail(userEmail)
                } catch {
                    errorMessage = "Failed to decode user data: \(error.localizedDescription)"
                    showingError = true
                    isJoining = false
                }
            }
        }.resume()
    }
    
    private func addMemberWithEmail(_ email: String) {
        guard let url = URL(string: "http://192.168.1.188:8000/projects/add_member") else {
            errorMessage = "Invalid URL"
            showingError = true
            isJoining = false
            return
        }
        
        var urlComponents = URLComponents(url: url, resolvingAgainstBaseURL: false)
        urlComponents?.queryItems = [
            URLQueryItem(name: "project_id", value: projectId),
            URLQueryItem(name: "user_id", value: userId),
            URLQueryItem(name: "new_email", value: email),
            URLQueryItem(name: "new_username", value: username.trimmingCharacters(in: .whitespacesAndNewlines)),
            URLQueryItem(name: "code", value: projectCode.trimmingCharacters(in: .whitespacesAndNewlines))
        ]
        
        guard let finalUrl = urlComponents?.url else {
            errorMessage = "Failed to create URL"
            showingError = true
            isJoining = false
            return
        }
        
        var request = URLRequest(url: finalUrl)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                isJoining = false
                
                if let error = error {
                    errorMessage = "Network error: \(error.localizedDescription)"
                    showingError = true
                    return
                }
                
                if let httpResponse = response as? HTTPURLResponse {
                    if httpResponse.statusCode == 200 {
                        onProjectJoined()
                        presentationMode.wrappedValue.dismiss()
                    } else {
                        errorMessage = "Failed to join project (Status: \(httpResponse.statusCode))"
                        showingError = true
                    }
                }
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
    @State private var selectedMembers: Set<String> = []
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Time Range")) {
                    DatePicker("Start Time", selection: $startDateTime, displayedComponents: [.date, .hourAndMinute])
                    DatePicker("End Time", selection: $endDateTime, displayedComponents: [.date, .hourAndMinute])
                }
                
                Section(header: Text("Members to Check")) {
                    ForEach(projectMembers, id: \.0) { member in
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(member.1) // Username
                                    .font(.body)
                                    .fontWeight(.medium)
                                Text(member.0) // Email
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            
                            Spacer()
                            
                            Button(action: {
                                if selectedMembers.contains(member.0) {
                                    selectedMembers.remove(member.0)
                                } else {
                                    selectedMembers.insert(member.0)
                                }
                            }) {
                                Image(systemName: selectedMembers.contains(member.0) ? "checkmark.circle.fill" : "circle")
                                    .foregroundColor(selectedMembers.contains(member.0) ? .blue : .gray)
                                    .font(.title2)
                            }
                        }
                        .contentShape(Rectangle())
                        .onTapGesture {
                            if selectedMembers.contains(member.0) {
                                selectedMembers.remove(member.0)
                            } else {
                                selectedMembers.insert(member.0)
                            }
                        }
                    }
                    
                    // Select All / Deselect All buttons
                    HStack {
                        Button("Select All") {
                            selectedMembers = Set(projectMembers.map { $0.0 })
                        }
                        .foregroundColor(.blue)
                        
                        Spacer()
                        
                        Button("Deselect All") {
                            selectedMembers.removeAll()
                        }
                        .foregroundColor(.red)
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
                    .disabled(isChecking || startDateTime >= endDateTime || selectedMembers.isEmpty)
                }
            }
            .alert("Error", isPresented: $showingError) {
                Button("OK") { }
            } message: {
                Text(errorMessage)
            }
            .onAppear {
                // Select all members by default
                selectedMembers = Set(projectMembers.map { $0.0 })
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
        guard let url = URL(string: "http://192.168.1.188:8000/coordinate/fetch_users") else {
            errorMessage = "Invalid URL"
            showingError = true
            isChecking = false
            return
        }
        
        // Filter project members to only include selected ones
        let selectedProjectMembers = projectMembers.filter { selectedMembers.contains($0.0) }
        
        // Convert selected project members (tuples) to array of dictionaries
        let membersArray = selectedProjectMembers.map { member in
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

        guard let url = URL(string: "http://192.168.1.188:8000/coordinate/get_availability") else {
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

// MARK: - Rename Project Sheet

struct RenameProjectSheet: View {
    let project: Project?
    @State var currentName: String
    let userId: String
    let onProjectRenamed: () -> Void
    
    @Environment(\.presentationMode) var presentationMode
    @State private var isRenaming = false
    @State private var errorMessage = ""
    @State private var showingError = false
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Project Name")) {
                    TextField("Enter new project name", text: $currentName)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                }
            }
            .navigationTitle("Rename Project")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        presentationMode.wrappedValue.dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Save") {
                        renameProject()
                    }
                    .disabled(isRenaming || currentName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
            }
            .alert("Error", isPresented: $showingError) {
                Button("OK") { }
            } message: {
                Text(errorMessage)
            }
        }
    }
    
    private func renameProject() {
        guard let project = project else {
            errorMessage = "No project selected"
            showingError = true
            return
        }
        
        isRenaming = true
        errorMessage = ""
        
        let trimmedName = currentName.trimmingCharacters(in: .whitespacesAndNewlines)

        guard let url = URL(string: "http://192.168.1.188:8000/projects/rename_project") else {
            errorMessage = "Invalid URL"
            showingError = true
            isRenaming = false
            return
        }
        
        let requestBody: [String: Any] = [
            "project_id": project.project_id,
            "user_id": userId,
            "project_name": trimmedName  // Backend expects project_name, not new_name
        ]
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            errorMessage = "Failed to encode request"
            showingError = true
            isRenaming = false
            return
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                isRenaming = false
                
                if let error = error {
                    errorMessage = "Network error: \(error.localizedDescription)"
                    showingError = true
                    return
                }
                
                if let httpResponse = response as? HTTPURLResponse {
                    guard (200...299).contains(httpResponse.statusCode) else {
                        errorMessage = "Server error: HTTP \(httpResponse.statusCode)"
                        showingError = true
                        return
                    }
                }
                
                // Close sheet and refresh projects
                presentationMode.wrappedValue.dismiss()
                onProjectRenamed()
            }
        }.resume()
    }
}

struct ProjectMemberManagementSheet: View {
    let projectId: String
    let userId: String
    let currentMembers: [(String, String)]
    let onMembersUpdated: () -> Void
    @Environment(\.presentationMode) var presentationMode
    
    @State private var members: [(String, String)]
    @State private var newMemberEmail = ""
    @State private var isLoading = false
    @State private var errorMessage = ""
    @State private var showingError = false
    @State private var showingPermissionEdit = false
    @State private var selectedMemberIndex: Int = 0
    
    init(projectId: String, userId: String, currentMembers: [(String, String)], onMembersUpdated: @escaping () -> Void) {
        self.projectId = projectId
        self.userId = userId
        self.currentMembers = currentMembers
        self.onMembersUpdated = onMembersUpdated
        self._members = State(initialValue: currentMembers)
    }
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Add member section
                VStack(alignment: .leading, spacing: 12) {
                    Text("Add New Member")
                        .font(.headline)
                        .foregroundColor(.primary)
                    
                    HStack {
                        TextField("Enter email address", text: $newMemberEmail)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                            .keyboardType(.emailAddress)
                            .autocapitalization(.none)
                        
                        Button(action: addMember) {
                            HStack {
                                Image(systemName: "plus.circle.fill")
                                Text("Add")
                            }
                            .foregroundColor(.white)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 8)
                            .background(Color.blue)
                            .cornerRadius(8)
                        }
                        .disabled(newMemberEmail.isEmpty || isLoading)
                    }
                }
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(12)
                
                // Current members section
                VStack(alignment: .leading, spacing: 12) {
                    Text("Current Members (\(members.count))")
                        .font(.headline)
                        .foregroundColor(.primary)
                    
                    if members.isEmpty {
                        Text("No members added yet")
                            .foregroundColor(.gray)
                            .italic()
                            .padding()
                    } else {
                        ScrollView {
                            LazyVStack(spacing: 8) {
                                ForEach(members.indices, id: \.self) { index in
                                    HStack {
                                        VStack(alignment: .leading, spacing: 4) {
                                            Text(members[index].0)
                                                .font(.subheadline)
                                                .fontWeight(.medium)
                                                .foregroundColor(.primary)
                                            
                                            if !members[index].1.isEmpty {
                                                Text(members[index].1)
                                                    .font(.caption)
                                                    .foregroundColor(.gray)
                                            }
                                        }
                                        
                                        Spacer()
                                        
                                        // Permissions button
                                        Button(action: {
                                            selectedMemberIndex = index
                                            showingPermissionEdit = true
                                        }) {
                                            Image(systemName: "person.badge.key")
                                                .foregroundColor(.blue)
                                                .font(.title3)
                                        }
                                        .disabled(isLoading)
                                        
                                        // Remove member button
                                        Button(action: {
                                            removeMember(at: index)
                                        }) {
                                            Image(systemName: "minus.circle.fill")
                                                .foregroundColor(.red)
                                                .font(.title2)
                                        }
                                        .disabled(isLoading)
                                    }
                                    .padding(.horizontal, 16)
                                    .padding(.vertical, 12)
                                    .background(Color(.systemBackground))
                                    .cornerRadius(8)
                                    .shadow(color: .gray.opacity(0.2), radius: 2, x: 0, y: 1)
                                }
                            }
                        }
                        .frame(maxHeight: 300)
                    }
                }
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(12)
                
                Spacer()
                
                if isLoading {
                    ProgressView("Updating members...")
                        .padding()
                }
            }
            .padding()
            .navigationTitle("Manage Members")
            .navigationBarTitleDisplayMode(.inline)
            .navigationBarItems(
                leading: Button("Cancel") {
                    presentationMode.wrappedValue.dismiss()
                },
                trailing: Button("Done") {
                    presentationMode.wrappedValue.dismiss()
                    onMembersUpdated()
                }
            )
        }
        .alert("Error", isPresented: $showingError) {
            Button("OK") { }
        } message: {
            Text(errorMessage)
        }
        .sheet(isPresented: $showingPermissionEdit) {
            if selectedMemberIndex < members.count {
                EditPermissionSheet(
                    projectId: projectId,
                    userId: userId,
                    memberEmail: members[selectedMemberIndex].0,
                    memberUsername: members[selectedMemberIndex].1,
                    onPermissionUpdated: onMembersUpdated
                )
            }
        }
    }
    
    private func addMember() {
        guard !newMemberEmail.isEmpty else { return }
        
        isLoading = true
        errorMessage = ""
        
        guard let url = URL(string: "http://192.168.1.188:8000/projects/add_member") else {
            errorMessage = "Invalid URL"
            showingError = true
            isLoading = false
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let requestBody = [
            "project_id": projectId,
            "member_email": newMemberEmail
        ]
        
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
                        // Add the new member to local state
                        members.append((newMemberEmail, ""))
                        newMemberEmail = ""
                    } else {
                        errorMessage = "Failed to add member (Status: \(httpResponse.statusCode))"
                        showingError = true
                    }
                }
            }
        }.resume()
    }
    
    private func removeMember(at index: Int) {
        let memberToRemove = members[index]
        
        isLoading = true
        errorMessage = ""
        
        guard let url = URL(string: "http://192.168.1.188:8000/projects/delete_member") else {
            errorMessage = "Invalid URL"
            showingError = true
            isLoading = false
            return
        }
        
        var urlComponents = URLComponents(url: url, resolvingAgainstBaseURL: false)
        urlComponents?.queryItems = [
            URLQueryItem(name: "project_id", value: projectId),
            URLQueryItem(name: "user_id", value: userId),
            URLQueryItem(name: "email", value: memberToRemove.0),
            URLQueryItem(name: "username", value: memberToRemove.1)
        ]
        
        guard let finalUrl = urlComponents?.url else {
            errorMessage = "Failed to create URL"
            showingError = true
            isLoading = false
            return
        }
        
        var request = URLRequest(url: finalUrl)
        request.httpMethod = "DELETE"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
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
                        // Remove the member from local state
                        members.remove(at: index)
                    } else {
                        errorMessage = "Failed to remove member (Status: \(httpResponse.statusCode))"
                        showingError = true
                    }
                }
            }
        }.resume()
    }
}

// MARK: - Edit Permission Sheet

struct EditPermissionSheet: View {
    let projectId: String
    let userId: String
    let memberEmail: String
    let memberUsername: String
    let onPermissionUpdated: () -> Void
    
    @Environment(\.presentationMode) var presentationMode
    @State private var selectedPermission: String = "view"
    @State private var isUpdating = false
    @State private var errorMessage = ""
    @State private var showingError = false
    
    private let permissions = ["view", "edit", "admin"]
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Member info
                VStack(alignment: .leading, spacing: 8) {
                    Text("Member Information")
                        .font(.headline)
                        .foregroundColor(.primary)
                    
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Email:")
                                .font(.subheadline)
                                .fontWeight(.medium)
                            Text(memberEmail)
                                .font(.subheadline)
                                .foregroundColor(.gray)
                        }
                        
                        if !memberUsername.isEmpty {
                            HStack {
                                Text("Username:")
                                    .font(.subheadline)
                                    .fontWeight(.medium)
                                Text(memberUsername)
                                    .font(.subheadline)
                                    .foregroundColor(.gray)
                            }
                        }
                    }
                }
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(12)
                
                // Permission selection
                VStack(alignment: .leading, spacing: 12) {
                    Text("Select Permission Level")
                        .font(.headline)
                        .foregroundColor(.primary)
                    
                    VStack(spacing: 8) {
                        ForEach(permissions, id: \.self) { permission in
                            Button(action: {
                                selectedPermission = permission
                            }) {
                                HStack {
                                    Image(systemName: selectedPermission == permission ? "checkmark.circle.fill" : "circle")
                                        .foregroundColor(selectedPermission == permission ? .blue : .gray)
                                        .font(.title2)
                                    
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text(permission.capitalized)
                                            .font(.subheadline)
                                            .fontWeight(.medium)
                                            .foregroundColor(.primary)
                                        
                                        Text(permissionDescription(for: permission))
                                            .font(.caption)
                                            .foregroundColor(.gray)
                                    }
                                    
                                    Spacer()
                                }
                                .padding()
                                .background(selectedPermission == permission ? Color.blue.opacity(0.1) : Color(.systemBackground))
                                .cornerRadius(8)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 8)
                                        .stroke(selectedPermission == permission ? Color.blue : Color.clear, lineWidth: 2)
                                )
                            }
                            .buttonStyle(PlainButtonStyle())
                        }
                    }
                }
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(12)
                
                Spacer()
                
                if isUpdating {
                    ProgressView("Updating permission...")
                        .padding()
                }
            }
            .padding()
            .navigationTitle("Edit Permission")
            .navigationBarTitleDisplayMode(.inline)
            .navigationBarItems(
                leading: Button("Cancel") {
                    presentationMode.wrappedValue.dismiss()
                },
                trailing: Button("Update") {
                    updatePermission()
                }
                .disabled(isUpdating)
            )
        }
        .alert("Error", isPresented: $showingError) {
            Button("OK") { }
        } message: {
            Text(errorMessage)
        }
    }
    
    private func permissionDescription(for permission: String) -> String {
        switch permission {
        case "view":
            return "Can view project content only"
        case "edit":
            return "Can view and edit project content"
        case "admin":
            return "Full access including member management"
        default:
            return ""
        }
    }
    
    private func updatePermission() {
        isUpdating = true
        errorMessage = ""
        
        guard let url = URL(string: "http://192.168.1.188:8000/projects/edit_permission") else {
            errorMessage = "Invalid URL"
            showingError = true
            isUpdating = false
            return
        }
        
        // Create URL with query parameters for email, username, and new_permission
        var urlComponents = URLComponents(url: url, resolvingAgainstBaseURL: false)
        urlComponents?.queryItems = [
            URLQueryItem(name: "email", value: memberEmail),
            URLQueryItem(name: "username", value: memberUsername),
            URLQueryItem(name: "new_permission", value: selectedPermission)
        ]
        
        guard let finalUrl = urlComponents?.url else {
            errorMessage = "Failed to create URL with parameters"
            showingError = true
            isUpdating = false
            return
        }
        
        // Create request body with only ModifyProjectRequest fields
        let requestBody: [String: Any] = [
            "project_id": projectId,
            "user_id": userId,
            "project_name": "" // Required by ModifyProjectRequest but not used
        ]
        
        var request = URLRequest(url: finalUrl)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            errorMessage = "Failed to encode request"
            showingError = true
            isUpdating = false
            return
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                isUpdating = false
                
                if let error = error {
                    errorMessage = "Network error: \(error.localizedDescription)"
                    showingError = true
                    return
                }
                
                if let httpResponse = response as? HTTPURLResponse {
                    print("Permission update response status: \(httpResponse.statusCode)")
                    
                    if let data = data {
                        if let responseString = String(data: data, encoding: .utf8) {
                            print("Permission update response: \(responseString)")
                        }
                    }
                    
                    if httpResponse.statusCode == 200 {
                        // Success
                        presentationMode.wrappedValue.dismiss()
                        onPermissionUpdated()
                    } else {
                        errorMessage = "Failed to update permission (Status: \(httpResponse.statusCode))"
                        if let data = data, let responseString = String(data: data, encoding: .utf8) {
                            errorMessage += ": \(responseString)"
                        }
                        showingError = true
                    }
                } else {
                    errorMessage = "Invalid response received"
                    showingError = true
                }
            }
        }.resume()
    }
}

struct ProjectsView_Previews: PreviewProvider {
    static var previews: some View {
        ProjectsView(userId: "preview-user-id")
    }
}

import SwiftUI

// MARK: - Data Models

struct Organization: Codable, Identifiable, Equatable {
    let id = UUID()
    let name: String
    let members: [String] // These are user IDs from the backend
    let projects: [String] // These are project IDs
    let organizationId: String // Backend ID
    
    private enum CodingKeys: String, CodingKey {
        case name
        case members
        case projects
        case organizationId = "id"
    }
    
    static func == (lhs: Organization, rhs: Organization) -> Bool {
        return lhs.organizationId == rhs.organizationId
    }
}

struct CreateOrganizationRequest: Codable {
    let id: String
    let name: String
    let members: [String]
    let projects: [String]
}

struct OrganizationRequest: Codable {
    let user_id: String
    let organization_id: String
}

// MARK: - Organizations Widget

struct OrganizationsWidget: View {
    let userId: String
    let projects: [Project] // Pass available projects for connecting
    let onOrganizationChanged: () -> Void
    
    @State private var organizations: [Organization] = []
    @State private var selectedOrganization: Organization? = nil
    @State private var isLoading = false
    @State private var errorMessage = ""
    @State private var showingError = false
    @State private var showingSlider = false
    
    // Sheet states
    @State private var showingCreateOrganization = false
    @State private var showingConnectProjects = false
    @State private var showingMemberManagement = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            HStack {
                Text("Organizations")
                    .font(.headline)
                    .fontWeight(.semibold)
                Spacer()
                Button(action: { fetchOrganizations() }) {
                    Image(systemName: "arrow.clockwise")
                        .foregroundColor(.blue)
                }
                .disabled(isLoading)
            }
            
            // Main content
            if isLoading {
                HStack {
                    ProgressView()
                        .scaleEffect(0.8)
                    Text("Loading organizations...")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.vertical, 8)
            } else if organizations.isEmpty {
                emptyStateView
            } else {
                organizationContent
            }
            
            // Action slider toggle
            Button(action: { showingSlider.toggle() }) {
                HStack {
                    Image(systemName: showingSlider ? "chevron.up" : "chevron.down")
                    Text("Organization Actions")
                        .font(.subheadline)
                }
                .foregroundColor(.blue)
            }
            
            // Expandable slider content
            if showingSlider {
                actionSlider
            }
        }
        .padding()
        .background(Color.clear)
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.white, lineWidth: 1)
        )
        .onAppear {
            fetchOrganizations()
        }
        .alert("Error", isPresented: $showingError) {
            Button("OK") { }
        } message: {
            Text(errorMessage)
        }
        .sheet(isPresented: $showingCreateOrganization) {
            CreateOrganizationSheet(
                userId: userId,
                onOrganizationCreated: {
                    fetchOrganizations()
                    onOrganizationChanged()
                }
            )
        }
        .sheet(isPresented: $showingConnectProjects) {
            if let organization = selectedOrganization {
                ConnectProjectsSheet(
                    userId: userId,
                    organization: organization,
                    availableProjects: projects,
                    onProjectsConnected: {
                        fetchOrganizations()
                        onOrganizationChanged()
                    }
                )
            }
        }
        .sheet(isPresented: $showingMemberManagement) {
            if let organization = selectedOrganization {
                MemberManagementSheet(
                    userId: userId,
                    organization: organization,
                    onMembersUpdated: {
                        fetchOrganizations()
                        onOrganizationChanged()
                    }
                )
            }
        }
    }
    
    private var emptyStateView: some View {
        VStack(spacing: 8) {
            Image(systemName: "building.2")
                .font(.title2)
                .foregroundColor(.secondary)
            Text("No organizations yet")
                .font(.subheadline)
                .foregroundColor(.secondary)
            Text("Create your first organization below")
                .font(.caption)
                .foregroundColor(Color(.tertiaryLabel))
        }
        .padding(.vertical, 12)
    }
    
    private var organizationContent: some View {
        HStack(spacing: 16) {
            // Dropdown for organizations
            Menu {
                ForEach(organizations) { org in
                    Button(org.name) {
                        selectedOrganization = org
                    }
                }
            } label: {
                HStack {
                    Text(selectedOrganization?.name ?? "Select Organization")
                        .font(.subheadline)
                        .foregroundColor(selectedOrganization != nil ? .primary : .secondary)
                    Spacer()
                    Image(systemName: "chevron.down")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(
                    RoundedRectangle(cornerRadius: 8)
                        .fill(Color(.systemBackground))
                        .stroke(Color(.systemGray4), lineWidth: 1)
                )
            }
            
            Spacer()
            
            // Project count
            if let selected = selectedOrganization {
                VStack(alignment: .trailing, spacing: 2) {
                    Text("\(selected.projects.count)")
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(.blue)
                    Text("Projects")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
    }
    
    private var actionSlider: some View {
        VStack(spacing: 12) {
            Divider()
            
            // Action buttons
            HStack(spacing: 16) {
                // Connect Projects
                ActionButton(
                    title: "Connect Projects",
                    icon: "link",
                    color: .blue
                ) {
                    if selectedOrganization != nil {
                        showingConnectProjects = true
                    }
                }
                
                // Create Organization
                ActionButton(
                    title: "New Organization",
                    icon: "plus.circle",
                    color: .green
                ) {
                    showingCreateOrganization = true
                }
                
                // Manage Members
                ActionButton(
                    title: "Manage Members",
                    icon: "person.2",
                    color: .orange
                ) {
                    if selectedOrganization != nil {
                        showingMemberManagement = true
                    }
                }
            }
            .padding(.top, 4)
        }
    }
    
    // MARK: - API Functions
    
    private func fetchOrganizations() {
        isLoading = true
        errorMessage = ""
        
    guard let url = URL(string: "http://192.168.1.188:8000/organizations/list_orgs?user_id=\(userId)") else {
            errorMessage = "Invalid URL"
            showingError = true
            isLoading = false
            return
        }
        
        URLSession.shared.dataTask(with: url) { data, response, error in
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
                    let fetchedOrganizations = try JSONDecoder().decode([Organization].self, from: data)
                    self.organizations = fetchedOrganizations
                    
                    // Auto-select first organization if none selected
                    if selectedOrganization == nil && !fetchedOrganizations.isEmpty {
                        selectedOrganization = fetchedOrganizations.first
                    }
                } catch {
                    errorMessage = "Failed to decode organizations: \(error.localizedDescription)"
                    showingError = true
                }
            }
        }.resume()
    }
}

// MARK: - Action Button Component

struct ActionButton: View {
    let title: String
    let icon: String
    let color: Color
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.title3)
                    .foregroundColor(color)
                
                Text(title)
                    .font(.caption)
                    .foregroundColor(.primary)
                    .multilineTextAlignment(.center)
                    .lineLimit(2)
            }
            .frame(maxWidth: .infinity, minHeight: 60)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(color.opacity(0.1))
            )
        }
        .buttonStyle(PlainButtonStyle())
    }
}

// MARK: - Create Organization Sheet

struct CreateOrganizationSheet: View {
    let userId: String
    let onOrganizationCreated: () -> Void
    
    @Environment(\.presentationMode) var presentationMode
    @State private var organizationName = ""
    @State private var isCreating = false
    @State private var errorMessage = ""
    @State private var showingError = false
    
    var body: some View {
        NavigationView {
            VStack(alignment: .leading, spacing: 20) {
                Text("Create New Organization")
                    .font(.title2)
                    .fontWeight(.bold)
                
                VStack(alignment: .leading, spacing: 8) {
                    Text("Organization Name")
                        .font(.subheadline)
                        .fontWeight(.medium)
                    
                    TextField("Enter organization name", text: $organizationName)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                }
                
                Spacer()
                
                Button(action: createOrganization) {
                    if isCreating {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            .scaleEffect(0.8)
                    } else {
                        Text("Create Organization")
                            .fontWeight(.semibold)
                    }
                }
                .frame(maxWidth: .infinity)
                .frame(height: 50)
                .background(organizationName.isEmpty ? Color.gray : Color.blue)
                .foregroundColor(.white)
                .cornerRadius(10)
                .disabled(organizationName.isEmpty || isCreating)
            }
            .padding()
            .navigationBarItems(
                leading: Button("Cancel") {
                    presentationMode.wrappedValue.dismiss()
                }
            )
        }
        .alert("Error", isPresented: $showingError) {
            Button("OK") { }
        } message: {
            Text(errorMessage)
        }
    }
    
    private func createOrganization() {
        isCreating = true
        errorMessage = ""
        
        let request = CreateOrganizationRequest(
            id: UUID().uuidString,
            name: organizationName,
            members: [userId],
            projects: []
        )

    guard let url = URL(string: "http://192.168.1.188:8000/organizations/create_org?user_id=\(userId)") else {
            errorMessage = "Invalid URL"
            showingError = true
            isCreating = false
            return
        }
        
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            urlRequest.httpBody = try JSONEncoder().encode(request)
        } catch {
            errorMessage = "Failed to encode request: \(error.localizedDescription)"
            showingError = true
            isCreating = false
            return
        }
        
        URLSession.shared.dataTask(with: urlRequest) { data, response, error in
            DispatchQueue.main.async {
                isCreating = false
                
                if let error = error {
                    errorMessage = "Network error: \(error.localizedDescription)"
                    showingError = true
                    return
                }
                
                if let httpResponse = response as? HTTPURLResponse {
                    if httpResponse.statusCode == 200 {
                        onOrganizationCreated()
                        presentationMode.wrappedValue.dismiss()
                    } else {
                        errorMessage = "Server error: \(httpResponse.statusCode)"
                        showingError = true
                    }
                }
            }
        }.resume()
    }
}

// MARK: - Connect Projects Sheet

struct ConnectProjectsSheet: View {
    let userId: String
    let organization: Organization
    let availableProjects: [Project]
    let onProjectsConnected: () -> Void
    
    @Environment(\.presentationMode) var presentationMode
    @State private var selectedProjects: Set<String> = []
    @State private var isConnecting = false
    @State private var errorMessage = ""
    @State private var showingError = false
    
    var unconnectedProjects: [Project] {
        availableProjects.filter { !organization.projects.contains($0.project_id) }
    }
    
    var body: some View {
        NavigationView {
            VStack(alignment: .leading, spacing: 20) {
                Text("Connect Projects to \(organization.name)")
                    .font(.title2)
                    .fontWeight(.bold)
                
                if unconnectedProjects.isEmpty {
                    VStack(spacing: 12) {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.title)
                            .foregroundColor(.green)
                        Text("All projects are already connected")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    Text("Select projects to connect:")
                        .font(.subheadline)
                        .fontWeight(.medium)
                    
                    ScrollView {
                        LazyVStack(spacing: 8) {
                            ForEach(unconnectedProjects) { project in
                                ProjectSelectionRow(
                                    project: project,
                                    isSelected: selectedProjects.contains(project.project_id)
                                ) {
                                    if selectedProjects.contains(project.project_id) {
                                        selectedProjects.remove(project.project_id)
                                    } else {
                                        selectedProjects.insert(project.project_id)
                                    }
                                }
                            }
                        }
                    }
                    
                    Button(action: connectProjects) {
                        if isConnecting {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                .scaleEffect(0.8)
                        } else {
                            Text("Connect \(selectedProjects.count) Project\(selectedProjects.count == 1 ? "" : "s")")
                                .fontWeight(.semibold)
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: 50)
                    .background(selectedProjects.isEmpty ? Color.gray : Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                    .disabled(selectedProjects.isEmpty || isConnecting)
                }
            }
            .padding()
            .navigationBarItems(
                leading: Button("Cancel") {
                    presentationMode.wrappedValue.dismiss()
                }
            )
        }
        .alert("Error", isPresented: $showingError) {
            Button("OK") { }
        } message: {
            Text(errorMessage)
        }
    }
    
    private func connectProjects() {
        isConnecting = true
        errorMessage = ""
        
        let group = DispatchGroup()
        var hasError = false
        
        for projectId in selectedProjects {
            guard let project = availableProjects.first(where: { $0.project_id == projectId }) else { continue }
            
            group.enter()
            
            let projectDict = [
                "project_id": project.project_id,
                "project_name": project.project_name,
                "project_members": project.project_members.map { $0.0 }, // Extract just the emails/IDs
                "project_likes": project.project_likes ?? 0,
                "project_transparency": project.project_transparency ?? true,
                "is_liked": project.is_liked ?? false
            ] as [String: Any]
            
            let requestBody = [
                "request": [
                    "user_id": userId,
                    "organization_id": organization.organizationId
                ],
                "project_dict": projectDict
            ] as [String: Any]
            
            guard let url = URL(string: "http://192.168.1.188:8000/organizations/add_project") else {
                hasError = true
                group.leave()
                continue
            }
            
            var urlRequest = URLRequest(url: url)
            urlRequest.httpMethod = "POST"
            urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
            
            do {
                urlRequest.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
            } catch {
                hasError = true
                group.leave()
                continue
            }
            
            URLSession.shared.dataTask(with: urlRequest) { data, response, error in
                defer { group.leave() }
                
                if error != nil || (response as? HTTPURLResponse)?.statusCode != 200 {
                    hasError = true
                }
            }.resume()
        }
        
        group.notify(queue: .main) {
            isConnecting = false
            
            if hasError {
                errorMessage = "Failed to connect some projects"
                showingError = true
            } else {
                onProjectsConnected()
                presentationMode.wrappedValue.dismiss()
            }
        }
    }
}

// MARK: - Project Selection Row

struct ProjectSelectionRow: View {
    let project: Project
    let isSelected: Bool
    let onTap: () -> Void
    
    var body: some View {
        Button(action: onTap) {
            HStack {
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .foregroundColor(isSelected ? .blue : .gray)
                
                VStack(alignment: .leading, spacing: 2) {
                    Text(project.project_name)
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(.primary)
                    
                    Text("\(project.project_members.count) member\(project.project_members.count == 1 ? "" : "s")")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(isSelected ? Color.blue.opacity(0.1) : Color(.systemGray6))
            )
        }
        .buttonStyle(PlainButtonStyle())
    }
}

// MARK: - Member Management Sheet

struct MemberManagementSheet: View {
    let userId: String
    let organization: Organization
    let onMembersUpdated: () -> Void
    
    @Environment(\.presentationMode) var presentationMode
    @State private var newMemberEmail = ""
    @State private var newMemberUsername = ""
    @State private var memberDetails: [(String, String)] = [] // [(userId, email)] pairs
    @State private var isLoading = false
    @State private var errorMessage = ""
    @State private var showingError = false
    
    var body: some View {
        NavigationView {
            VStack(alignment: .leading, spacing: 20) {
                Text("Manage Members")
                    .font(.title2)
                    .fontWeight(.bold)
                
                Text("Organization: \(organization.name)")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                
                // Add new member section
                VStack(alignment: .leading, spacing: 12) {
                    Text("Add New Member")
                        .font(.headline)
                    
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Email")
                            .font(.subheadline)
                            .fontWeight(.medium)
                        TextField("Enter email address", text: $newMemberEmail)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                            .keyboardType(.emailAddress)
                            .autocapitalization(.none)
                    }
                    
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Username")
                            .font(.subheadline)
                            .fontWeight(.medium)
                        TextField("Enter username", text: $newMemberUsername)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                            .autocapitalization(.none)
                    }
                    
                    Button(action: addMember) {
                        if isLoading {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                .scaleEffect(0.8)
                        } else {
                            Text("Add Member")
                                .fontWeight(.semibold)
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: 44)
                    .background(canAddMember ? Color.green : Color.gray)
                    .foregroundColor(.white)
                    .cornerRadius(8)
                    .disabled(!canAddMember || isLoading)
                }
                
                Divider()
                
                // Current members section
                VStack(alignment: .leading, spacing: 12) {
                    Text("Current Members (\(memberDetails.count))")
                        .font(.headline)
                    
                    if memberDetails.isEmpty {
                        Text("No members found")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .frame(maxWidth: .infinity, alignment: .center)
                            .padding(.vertical, 20)
                    } else {
                        ScrollView {
                            LazyVStack(spacing: 8) {
                                ForEach(Array(memberDetails.enumerated()), id: \.offset) { index, member in
                                    MemberRow(
                                        memberId: member.0,
                                        memberEmail: member.1,
                                        canDelete: member.0 != userId
                                    ) {
                                        removeMember(member.1) // Pass email for API
                                    }
                                }
                            }
                        }
                    }
                }
                
                Spacer()
            }
            .padding()
            .navigationBarItems(
                leading: Button("Done") {
                    presentationMode.wrappedValue.dismiss()
                }
            )
        }
        .onAppear {
            // For now, we'll use member IDs as both ID and email
            // In a real implementation, you'd need to fetch user details
            memberDetails = organization.members.map { ($0, $0) }
        }
        .alert("Error", isPresented: $showingError) {
            Button("OK") { }
        } message: {
            Text(errorMessage)
        }
    }
    
    private var canAddMember: Bool {
        !newMemberEmail.isEmpty && !newMemberUsername.isEmpty && newMemberEmail.contains("@")
    }
    
    private func addMember() {
        isLoading = true
        errorMessage = ""
        
        let orgRequest = OrganizationRequest(
            user_id: userId,
            organization_id: organization.organizationId
        )

    guard let url = URL(string: "http://192.168.1.188:8000/organizations/add_member?new_email=\(newMemberEmail)&new_username=\(newMemberUsername)") else {
            errorMessage = "Invalid URL"
            showingError = true
            isLoading = false
            return
        }
        
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            urlRequest.httpBody = try JSONEncoder().encode(orgRequest)
        } catch {
            errorMessage = "Failed to encode request: \(error.localizedDescription)"
            showingError = true
            isLoading = false
            return
        }
        
        URLSession.shared.dataTask(with: urlRequest) { data, response, error in
            DispatchQueue.main.async {
                isLoading = false
                
                if let error = error {
                    errorMessage = "Network error: \(error.localizedDescription)"
                    showingError = true
                    return
                }
                
                if let httpResponse = response as? HTTPURLResponse {
                    if httpResponse.statusCode == 200 {
                        let emailToAdd = newMemberEmail
                        newMemberEmail = ""
                        newMemberUsername = ""
                        // Refresh member details - in a real app you'd refetch organization details
                        // For now, just add the new member
                        memberDetails.append((emailToAdd, emailToAdd))
                        onMembersUpdated()
                    } else {
                        errorMessage = "Failed to add member. Please check the email and username."
                        showingError = true
                    }
                }
            }
        }.resume()
    }
    
    private func removeMember(_ email: String) {
        let orgRequest = OrganizationRequest(
            user_id: userId,
            organization_id: organization.organizationId
        )

    guard let url = URL(string: "http://192.168.1.188:8000/organizations/remove_member?email=\(email)") else {
            errorMessage = "Invalid URL"
            showingError = true
            return
        }
        
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "DELETE"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            urlRequest.httpBody = try JSONEncoder().encode(orgRequest)
        } catch {
            errorMessage = "Failed to encode request: \(error.localizedDescription)"
            showingError = true
            return
        }
        
        URLSession.shared.dataTask(with: urlRequest) { data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    errorMessage = "Network error: \(error.localizedDescription)"
                    showingError = true
                    return
                }
                
                if let httpResponse = response as? HTTPURLResponse {
                    if httpResponse.statusCode == 200 {
                        memberDetails.removeAll { $0.1 == email }
                        onMembersUpdated()
                    } else {
                        errorMessage = "Failed to remove member"
                        showingError = true
                    }
                }
            }
        }.resume()
    }
}

// MARK: - Member Row

struct MemberRow: View {
    let memberId: String
    let memberEmail: String
    let canDelete: Bool
    let onDelete: () -> Void
    
    var body: some View {
        HStack {
            Image(systemName: "person.circle.fill")
                .foregroundColor(.blue)
                .font(.title2)
            
            VStack(alignment: .leading, spacing: 2) {
                Text(memberEmail)
                    .font(.subheadline)
                    .fontWeight(.medium)
                Text("Member")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            if canDelete {
                Button(action: onDelete) {
                    Image(systemName: "minus.circle.fill")
                        .foregroundColor(.red)
                        .font(.title3)
                }
            } else {
                Text("You")
                    .font(.caption)
                    .foregroundColor(.blue)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.blue.opacity(0.1))
                    .cornerRadius(4)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color(.systemGray6))
        )
    }
}

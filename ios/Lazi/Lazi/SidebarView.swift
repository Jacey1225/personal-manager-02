import SwiftUI

struct SidebarView: View {
    @Binding var isPresented: Bool
    let userId: String  // Add userId parameter
    
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            VStack(alignment: .leading, spacing: 8) {
                Text("Lazi")
                    .font(.system(size: 24, weight: .bold, design: .rounded))
                    .foregroundColor(.white)
                
                Text("Personal Manager")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundColor(.white.opacity(0.7))
                
                // Optional: Show user ID (first 8 characters for brevity)
                Text("User: \(String(userId.prefix(8)))...")
                    .font(.system(size: 10, weight: .light))
                    .foregroundColor(.white.opacity(0.5))
            }
            .padding(.horizontal, 20)
            .padding(.top, 20)
            .padding(.bottom, 30)
            
            // Navigation Items
            VStack(alignment: .leading, spacing: 0) {
                NavigationLink(destination: HomePage(userId: userId)) {
                    SidebarRow(icon: "house.fill", title: "Home", isSelected: true)
                }
                .buttonStyle(PlainButtonStyle())
                
                NavigationLink(destination: CalendarView(userId: userId)) {
                    SidebarRow(icon: "calendar", title: "Calendar")
                }
                .buttonStyle(PlainButtonStyle())
                
                NavigationLink(destination: TasksListView(userId: userId)) {
                    SidebarRow(icon: "checklist", title: "Tasks")
                }
                .buttonStyle(PlainButtonStyle())
                
                NavigationLink(destination: ProjectsView(userId: userId)) {
                    SidebarRow(icon: "folder", title: "Projects")
                }
                .buttonStyle(PlainButtonStyle())
                
                NavigationLink(destination: SettingsView(userId: userId)) {
                    SidebarRow(icon: "gearshape.fill", title: "Settings")
                }
                .buttonStyle(PlainButtonStyle())
            }
            .padding(.top, 60) // Add padding to avoid overlap with sidebar button
            
            Spacer()
            
            // Footer
            VStack(alignment: .leading, spacing: 8) {
                Divider()
                    .background(Color.white.opacity(0.3))
                
                Text("v1.0.0")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(.white.opacity(0.5))
                    .padding(.horizontal, 20)
                    .padding(.bottom, 20)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
        .background(Color.black.opacity(0.9))
        .ignoresSafeArea()
    }
}

struct SidebarRow: View {
    let icon: String
    let title: String
    var isSelected: Bool = false
    
    var body: some View {
        HStack(spacing: 15) {
            Image(systemName: icon)
                .font(.system(size: 18, weight: .medium))
                .foregroundColor(isSelected ? .white : .white.opacity(0.7))
                .frame(width: 24)
            
            Text(title)
                .font(.system(size: 16, weight: .medium))
                .foregroundColor(isSelected ? .white : .white.opacity(0.7))
            
            Spacer()
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .background(
            Rectangle()
                .fill(isSelected ? Color.white.opacity(0.1) : Color.clear)
        )
        .contentShape(Rectangle())
    }
}

// MARK: - Placeholder Views for Navigation

struct CalendarView: View {
    let userId: String
    
    var body: some View {
        VStack {
            Text("Calendar View")
                .font(.largeTitle)
                .padding()
            
            Text("User ID: \(userId)")
                .font(.caption)
                .foregroundColor(.secondary)
            
            Text("Calendar functionality coming soon...")
                .foregroundColor(.secondary)
        }
        .navigationTitle("Calendar")
        .navigationBarTitleDisplayMode(.inline)
    }
}

struct SettingsView: View {
    let userId: String
    @State private var showingDeleteConfirmation = false
    @State private var showingDeleteSuccess = false
    @State private var isDeleting = false
    @State private var showingDeleteProjectsConfirmation = false
    @State private var showingDeleteProjectsSuccess = false
    @State private var isDeletingProjects = false
    @State private var errorMessage = ""
    @State private var showingError = false
    @Environment(\.presentationMode) var presentationMode
    
    var body: some View {
        VStack(spacing: 24) {
            // Header
            VStack(spacing: 8) {
                Image(systemName: "gearshape.fill")
                    .font(.system(size: 48))
                    .foregroundColor(.blue)
                
                Text("Settings")
                    .font(.largeTitle)
                    .fontWeight(.bold)
            }
            .padding(.top, 20)
            
            // User Info Section
            VStack(alignment: .leading, spacing: 12) {
                Text("Account Information")
                    .font(.headline)
                    .foregroundColor(.primary)
                
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("User ID:")
                            .font(.subheadline)
                            .fontWeight(.medium)
                        Spacer()
                        Text(String(userId.prefix(12)) + "...")
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color(.systemGray5))
                            .cornerRadius(6)
                    }
                }
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(12)
            }
            .padding(.horizontal)
            
            Spacer()
            
            // Danger Zone
            VStack(alignment: .leading, spacing: 16) {
                Text("Danger Zone")
                    .font(.headline)
                    .foregroundColor(.red)
                
                // Delete All Projects Section
                VStack(spacing: 12) {
                    Text("Delete All Projects")
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(.primary)
                    
                    Text("This will permanently delete all your projects. This action cannot be undone.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                    
                    if isDeletingProjects {
                        ProgressView("Deleting projects...")
                            .padding()
                    } else {
                        Button(action: {
                            showingDeleteProjectsConfirmation = true
                        }) {
                            HStack {
                                Image(systemName: "folder.badge.minus")
                                Text("Delete All Projects")
                            }
                            .foregroundColor(.white)
                            .padding(.horizontal, 24)
                            .padding(.vertical, 12)
                            .background(Color.orange)
                            .cornerRadius(8)
                        }
                    }
                }
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(12)
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.orange.opacity(0.3), lineWidth: 1)
                )
                
                // Delete Account Section
                VStack(spacing: 12) {
                    Text("Delete Account")
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(.primary)
                    
                    Text("This action cannot be undone. All your data will be permanently deleted.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                    
                    if isDeleting {
                        ProgressView("Deleting account...")
                            .padding()
                    } else {
                        Button(action: {
                            showingDeleteConfirmation = true
                        }) {
                            HStack {
                                Image(systemName: "trash")
                                Text("Delete Account")
                            }
                            .foregroundColor(.white)
                            .padding(.horizontal, 24)
                            .padding(.vertical, 12)
                            .background(Color.red)
                            .cornerRadius(8)
                        }
                    }
                }
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(12)
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.red.opacity(0.3), lineWidth: 1)
                )
            }
            .padding(.horizontal)
            .padding(.bottom, 40)
        }
        .navigationTitle("Settings")
        .navigationBarTitleDisplayMode(.inline)
        .alert("Delete Account", isPresented: $showingDeleteConfirmation) {
            Button("Cancel", role: .cancel) { }
            Button("Delete", role: .destructive) {
                deleteAccount()
            }
        } message: {
            Text("Are you sure you want to delete your account? This action cannot be undone and all your data will be permanently lost.")
        }
        .alert("Delete All Projects", isPresented: $showingDeleteProjectsConfirmation) {
            Button("Cancel", role: .cancel) { }
            Button("Delete", role: .destructive) {
                deleteAllProjects()
            }
        } message: {
            Text("Are you sure you want to delete all your projects? This action cannot be undone.")
        }
        .alert("Account Deleted", isPresented: $showingDeleteSuccess) {
            Button("OK") {
                // Navigate back to login or close app
                presentationMode.wrappedValue.dismiss()
            }
        } message: {
            Text("Your account has been successfully deleted.")
        }
        .alert("Projects Deleted", isPresented: $showingDeleteProjectsSuccess) {
            Button("OK") { }
        } message: {
            Text("All your projects have been successfully deleted.")
        }
        .alert("Error", isPresented: $showingError) {
            Button("OK") { }
        } message: {
            Text(errorMessage)
        }
    }
    
    private func deleteAccount() {
        isDeleting = true
        errorMessage = ""
        
        guard let url = URL(string: "https://29098e308ec4.ngrok-free.app/auth/remove_user") else {
            errorMessage = "Invalid URL"
            showingError = true
            isDeleting = false
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let requestBody = ["user_id": userId]
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            errorMessage = "Failed to encode request"
            showingError = true
            isDeleting = false
            return
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                isDeleting = false
                
                if let error = error {
                    errorMessage = "Network error: \(error.localizedDescription)"
                    showingError = true
                    return
                }
                
                if let httpResponse = response as? HTTPURLResponse {
                    if httpResponse.statusCode == 200 {
                        showingDeleteSuccess = true
                    } else {
                        errorMessage = "Failed to delete account (Status: \(httpResponse.statusCode))"
                        showingError = true
                    }
                } else {
                    errorMessage = "Invalid response from server"
                    showingError = true
                }
            }
        }.resume()
    }
    
    private func deleteAllProjects() {
        isDeletingProjects = true
        errorMessage = ""
        
        guard let url = URL(string: "https://29098e308ec4.ngrok-free.app/projects/global_delete") else {
            errorMessage = "Invalid URL"
            showingError = true
            isDeletingProjects = false
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let requestBody = [
            "project_id": "",
            "user_id": userId,
            "project_name": ""
        ]
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            errorMessage = "Failed to encode request"
            showingError = true
            isDeletingProjects = false
            return
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                isDeletingProjects = false
                
                if let error = error {
                    errorMessage = "Network error: \(error.localizedDescription)"
                    showingError = true
                    return
                }
                
                if let httpResponse = response as? HTTPURLResponse {
                    if httpResponse.statusCode == 200 {
                        showingDeleteProjectsSuccess = true
                    } else {
                        errorMessage = "Failed to delete projects (Status: \(httpResponse.statusCode))"
                        showingError = true
                    }
                } else {
                    errorMessage = "Invalid response from server"
                    showingError = true
                }
            }
        }.resume()
    }
}

struct SidebarView_Previews: PreviewProvider {
    static var previews: some View {
        SidebarView(isPresented: .constant(true), userId: "preview-user-id")
    }
}

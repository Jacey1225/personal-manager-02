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
    
    var body: some View {
        VStack {
            Text("Settings View")
                .font(.largeTitle)
                .padding()
            
            Text("User ID: \(userId)")
                .font(.caption)
                .foregroundColor(.secondary)
            
            Text("Settings functionality coming soon...")
                .foregroundColor(.secondary)
        }
        .navigationTitle("Settings")
        .navigationBarTitleDisplayMode(.inline)
    }
}

struct SidebarView_Previews: PreviewProvider {
    static var previews: some View {
        SidebarView(isPresented: .constant(true), userId: "preview-user-id")
    }
}

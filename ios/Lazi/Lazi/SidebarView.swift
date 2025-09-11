import SwiftUI

struct SidebarView: View {
    @Binding var isPresented: Bool
    
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
            }
            .padding(.horizontal, 20)
            .padding(.top, 20)
            .padding(.bottom, 30)
            
            // Navigation Items
            VStack(alignment: .leading, spacing: 0) {
                NavigationLink(destination: HomePage()) {
                    SidebarRow(icon: "house.fill", title: "Home", isSelected: true)
                }
                .buttonStyle(PlainButtonStyle())
                
                NavigationLink(destination: Text("Calendar View").foregroundColor(.white)) {
                    SidebarRow(icon: "calendar", title: "Calendar")
                }
                .buttonStyle(PlainButtonStyle())
                
                NavigationLink(destination: Text("Tasks View").foregroundColor(.white)) {
                    SidebarRow(icon: "checklist", title: "Tasks")
                }
                .buttonStyle(PlainButtonStyle())
                
                NavigationLink(destination: Text("Settings View").foregroundColor(.white)) {
                    SidebarRow(icon: "gearshape.fill", title: "Settings")
                }
                .buttonStyle(PlainButtonStyle())
            }
            
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

struct SidebarView_Previews: PreviewProvider {
    static var previews: some View {
        SidebarView(isPresented: .constant(true))
    }
}

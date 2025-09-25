import SwiftUI

struct HomePage: View {
    let userId: String
    
    // Sidebar state
    @State private var showSidebar = false

    var body: some View {
        ZStack {
            NavigationView {
                VStack(spacing: 16) {
                    Spacer()
                    
                    Text("Welcome to Lazi")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                    
                    Text("Your personal task manager")
                        .font(.headline)
                        .foregroundColor(.secondary)
                    
                    Spacer()
                }
                .navigationBarHidden(true)
                .padding()
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
}

struct HomePage_Previews: PreviewProvider {
    static var previews: some View {
        HomePage(userId: "preview-user-id")
    }
}
import SwiftUI

struct ContentView: View {
    @State private var isNavigating: Bool = false
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
            Image(systemName: "globe")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Text("Hello, Jacey!")
            Text("I am Lazi, your personal scheduler.")

            Button(action: {
                withAnimation(.easeInOut(duration: 0.5)) {
                    isNavigating = true
                }
            }) {
                Text("Dive in!")
                    .font(.headline)
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }

            NavigationLink(destination: HomePage(), isActive: $isNavigating) {
                EmptyView()
            }
        }
        .padding()
        }
    }
}

#Preview {
    ContentView()
}

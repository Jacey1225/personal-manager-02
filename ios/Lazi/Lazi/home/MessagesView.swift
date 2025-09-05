import SwiftUI

struct ChatMessage: Identifiable {
    let id = UUID()
    let text: String
}

struct ChatMessagesView: View {
    let messages: [ChatMessage]
    var body: some View {
        List(messages) { message in 
            HStack {
                Text(message.text)
                    .padding(10)
                    .background(Color.white.opacity(0.8))
                    .foregroundColor(.black)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                    .frame(maxWidth: .infinity, alignment: .trailing)
                Spacer()
            }
            .listRowBackground(Color.clear)
            .listRowSeparator(.hidden)
        }
        .listStyle(.plain)
        .padding(.horizontal)
    }
}

struct ChatMessagesView_Preview: PreviewProvider {
    static var previews: some View {
        ChatMessagesView(messages: [
            ChatMessage(text: "Hello! How can I assist you today?"),
            ChatMessage(text: "I need help scheduling a meeting."),
            ChatMessage(text: "Sure! When would you like to schedule it?")
        ])
        .background(
            LinearGradient(gradient: Gradient(colors: [Color.blue.opacity(0.3), Color.purple.opacity(0.3)]), startPoint: .topLeading, endPoint: .bottomTrailing)
                .ignoresSafeArea()
        )
    }
}
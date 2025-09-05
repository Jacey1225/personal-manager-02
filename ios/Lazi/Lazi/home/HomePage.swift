import SwiftUI
import CoreHaptics

struct HomePage: View {
    @State private var userInput: String = ""
    @State private var responseMessage: String = ""
    @State private var messages: [ChatMessage] = []
    @State private var buttonTapped: Bool = false

    var body: some View {
        VStack {
            ChatMessagesView(messages: messages)
            HStack {
                ZStack(alignment: .trailing) {
                    TextField("Give me a task!", text: $userInput)
                    .font(.system(size: 16, weight: .medium, design: .rounded))
                    .foregroundColor(.black)
                    .padding(10)
                    .padding(.trailing, 40)
                    .background(Color.white)
                    .cornerRadius(20)
                    .overlay(
                        RoundedRectangle(cornerRadius: 20)
                            .stroke(Color.white, lineWidth: 1)
                    )
                    .padding(.horizontal, 20)
                    .padding(.bottom, 10)
                    Button(action: {
                        sendToAPI()
                    }) {
                        Image(systemName: "paperplane.fill")
                            .foregroundColor(.gray)
                            .padding(10)
                    }     
                    .padding(.trailing, 25)    
                    .padding(.bottom, 10)           
                }

                Button(action: {
                    let haptic = UIImpactFeedbackGenerator(style: .light)
                    haptic.impactOccurred()
                    withAnimation(.easeInOut(duration: 0.2)) {
                        buttonTapped = true
                    }
                    sendToAPI()
                    withAnimation(.easeInOut(duration: 0.2).delay(0.2)) {
                        buttonTapped = false
                    }
                }) {
                    Image(systemName: "speaker.wave.2.fill")
                        .foregroundColor(.gray)
                        .padding(10)
                        .background(Color.white)
                        .clipShape(Circle())
                }
                .padding(.bottom, 10)
            }
        }
        .padding()
        .navigationTitle("Chat with Lazi")
        .navigationBarTitleDisplayMode(.inline)
        .ignoresSafeArea(.keyboard, edges: .bottom)
        .background(
        LinearGradient(gradient: Gradient(colors: [Color.blue.opacity(0.3), Color.purple.opacity(1)]), startPoint: .topLeading, endPoint: .bottomTrailing)
            .edgesIgnoringSafeArea(.all)
        )
    } 

    func sendToAPI() {
        guard !userInput.isEmpty else {
            responseMessage = "Please enter a task."
            return
        }
        messages.append(ChatMessage(text: userInput))
    }
}

struct HomePage_Previews: PreviewProvider {
    static var previews: some View {
        HomePage()
    }
}
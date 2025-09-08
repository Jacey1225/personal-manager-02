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
        
        // Add user message (always true for user input)
        messages.append(ChatMessage(text: userInput, isUserMessage: true))
        let currentInput = userInput
        userInput = ""

        guard let url = URL(string: "http://localhost:8000/scheduler/process_input") else {
            messages.append(ChatMessage(text: "Invalid URL.", isUserMessage: false))
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let requestBody = ["input_text": currentInput]
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            messages.append(ChatMessage(text: "Failed to encode request.", isUserMessage: false))
            return
        }

        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    self.messages.append(ChatMessage(text: "Error: \(error.localizedDescription)", isUserMessage: false))
                    return
                }

                guard let data = data else {
                    self.messages.append(ChatMessage(text: "No data received.", isUserMessage: false))
                    return
                }

                do {
                    if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                       let message = json["message"] as? String {
                        // System response (always false for API responses)
                        self.messages.append(ChatMessage(text: message, isUserMessage: false))
                    } else {
                        let rawResponse = String(data: data, encoding: .utf8) ?? "Unknown response"
                        self.messages.append(ChatMessage(text: rawResponse, isUserMessage: false))
                    }
                } catch {
                    let rawResponse = String(data: data, encoding: .utf8) ?? "Failed to parse response"
                    self.messages.append(ChatMessage(text: rawResponse, isUserMessage: false))
                }
            }
        }
        task.resume()
    }
}

struct HomePage_Previews: PreviewProvider {
    static var previews: some View {
        HomePage()
    }
}
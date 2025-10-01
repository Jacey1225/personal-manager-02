import SwiftUI
import SafariServices

struct User {
    let userId: String
    let username: String
    let email: String
}

struct ContentView: View {
    @State private var isNavigating: Bool = false
    @State private var showingLogin = false
    @State private var showingSignup = false
    @State private var showingGoogleAuth = false
    @State private var currentUser: User?
    @State private var isAuthenticated = false
    @State private var googleAuthCompleted = false
    @State private var authURL: String = ""
    @State private var errorMessage = ""
    @State private var isLoading = false
    
    // Login/Signup form fields
    @State private var username = ""
    @State private var email = ""
    @State private var password = ""
    
    // Add this state variable for the authorization code
    @State private var authCode = ""

    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                if !isAuthenticated {
                    // Welcome Screen
                    welcomeView
                } else if !googleAuthCompleted {
                    // Google Authentication Screen
                    googleAuthView
                } else {
                    // Success Screen
                    successView
                }
            }
            .padding()
            .sheet(isPresented: $showingLogin) {
                loginView
            }
            .sheet(isPresented: $showingSignup) {
                signupView
            }
            .sheet(isPresented: $showingGoogleAuth) {
                NavigationView {
                    VStack(spacing: 20) {
                        Text("Complete Google Authentication")
                            .font(.title2)
                            .fontWeight(.semibold)
                        
                        Text("1. Tap 'Open Google Auth' to open the authorization page")
                            .multilineTextAlignment(.center)
                        Text("2. Sign in and authorize the app")
                            .multilineTextAlignment(.center)
                        Text("3. Copy the authorization code")
                            .multilineTextAlignment(.center)
                        Text("4. Paste it below and tap 'Complete'")
                            .multilineTextAlignment(.center)
                        
                        Button("Open Google Auth") {
                            if let url = URL(string: authURL) {
                                UIApplication.shared.open(url)
                            }
                        }
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                        
                        TextField("Authorization Code", text: $authCode)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                            .font(.system(.body, design: .monospaced))
                        
                        if isLoading {
                            ProgressView("Completing authentication...")
                                .padding()
                        } else {
                            Button("Complete Authentication") {
                                completeGoogleAuth()
                            }
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(authCode.isEmpty ? Color.gray : Color.green)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                            .disabled(authCode.isEmpty)
                        }
                        
                        Button("Cancel") {
                            showingGoogleAuth = false
                            authCode = ""
                        }
                        .foregroundColor(.red)
                    }
                    .padding()
                    .navigationBarTitleDisplayMode(.inline)
                    .navigationBarHidden(true)
                }
            }
            .alert("Error", isPresented: .constant(!errorMessage.isEmpty)) {
                Button("OK") {
                    errorMessage = ""
                }
            } message: {
                Text(errorMessage)
            }
        }
    }
    
    // MARK: - View Components
    
    var welcomeView: some View {
        VStack(spacing: 30) {
            Image(systemName: "globe")
                .imageScale(.large)
                .foregroundStyle(.tint)
                .font(.system(size: 60))
            
            VStack(spacing: 10) {
                Text("Welcome to Lazi")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                
                Text("Your Personal Scheduler")
                    .font(.title2)
                    .foregroundColor(.secondary)
            }
            
            VStack(spacing: 15) {
                Button(action: {
                    showingLogin = true
                }) {
                    Text("Log In")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }
                
                Button(action: {
                    showingSignup = true
                }) {
                    Text("Sign Up")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.green)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }
            }
            .padding(.horizontal)
        }
    }
    
    var googleAuthView: some View {
        VStack(spacing: 30) {
            Image(systemName: "lock.shield")
                .imageScale(.large)
                .foregroundStyle(.tint)
                .font(.system(size: 60))
            
            VStack(spacing: 15) {
                Text("Connect Google Account")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                    .multilineTextAlignment(.center)
                
                Text("To access your calendar and tasks, we need permission to connect to your Google account.")
                    .font(.body)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }
            
            if isLoading {
                ProgressView("Setting up Google authentication...")
                    .padding()
            } else {
                Button(action: {
                    authenticateWithGoogle()
                }) {
                    HStack {
                        Image(systemName: "globe")
                        Text("Connect Google Account")
                    }
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.red)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                }
                .padding(.horizontal)
            }
        }
    }
    
    var successView: some View {
        VStack(spacing: 30) {
            Image(systemName: "checkmark.circle.fill")
                .imageScale(.large)
                .foregroundColor(.green)
                .font(.system(size: 60))
            
            VStack(spacing: 10) {
                Text("Hello, \(currentUser?.username ?? "User")!")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                
                Text("I am Lazi, your personal scheduler.")
                    .font(.title2)
                    .foregroundColor(.secondary)
            }
            
            Button(action: {
                withAnimation(.easeInOut(duration: 0.5)) {
                    isNavigating = true
                }
            }) {
                Text("Dive in!")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }
            .padding(.horizontal)
            
            NavigationLink(
                destination: HomePage(userId: currentUser?.userId ?? ""),
                isActive: $isNavigating
            ) {
                EmptyView()
            }
        }
    }
    
    var loginView: some View {
        NavigationView {
            VStack(spacing: 20) {
                Text("Log In")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                    .padding(.top)
                
                VStack(spacing: 15) {
                    TextField("Username", text: $username)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .autocapitalization(.none)
                    
                    SecureField("Password", text: $password)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                }
                .padding(.horizontal)
                
                if isLoading {
                    ProgressView("Logging in...")
                        .padding()
                } else {
                    Button(action: {
                        login()
                    }) {
                        Text("Log In")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }
                    .padding(.horizontal)
                    .disabled(username.isEmpty || password.isEmpty)
                }
                
                Spacer()
            }
            .navigationBarItems(
                trailing: Button("Cancel") {
                    showingLogin = false
                    clearForm()
                }
            )
        }
    }
    
    var signupView: some View {
        NavigationView {
            VStack(spacing: 20) {
                Text("Sign Up")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                    .padding(.top)
                
                VStack(spacing: 15) {
                    TextField("Username", text: $username)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .autocapitalization(.none)
                    
                    TextField("Email", text: $email)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .autocapitalization(.none)
                        .keyboardType(.emailAddress)
                    
                    SecureField("Password", text: $password)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                }
                .padding(.horizontal)
                
                if isLoading {
                    ProgressView("Creating account...")
                        .padding()
                } else {
                    Button(action: {
                        signup()
                    }) {
                        Text("Sign Up")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.green)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }
                    .padding(.horizontal)
                    .disabled(username.isEmpty || email.isEmpty || password.isEmpty)
                }
                
                Spacer()
            }
            .navigationBarItems(
                trailing: Button("Cancel") {
                    showingSignup = false
                    clearForm()
                }
            )
        }
    }
    
    // MARK: - API Functions
    
    func login() {
        guard !username.isEmpty && !password.isEmpty else {
            errorMessage = "Please fill in all fields"
            return
        }
        
        isLoading = true

    guard let url = URL(string: "http://192.168.1.222:8000/auth/login?username=\(username)&password=\(password)") else {
            errorMessage = "Invalid URL"
            isLoading = false
            return
        }
        
        URLSession.shared.dataTask(with: url) { data, response, error in
            DispatchQueue.main.async {
                isLoading = false
                
                if let error = error {
                    errorMessage = "Network error: \(error.localizedDescription)"
                    return
                }
                
                guard let data = data else {
                    errorMessage = "No data received"
                    return
                }
                
                do {
                    if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                        if let status = json["status"] as? String {
                            if status == "success", let userId = json["user_id"] as? String {
                                currentUser = User(userId: userId, username: username, email: "")
                                isAuthenticated = true
                                showingLogin = false
                                clearForm()
                            } else {
                                errorMessage = json["message"] as? String ?? "Login failed"
                            }
                        }
                    }
                } catch {
                    errorMessage = "Failed to parse response"
                }
            }
        }.resume()
    }
    
    func signup() {
        guard !username.isEmpty && !email.isEmpty && !password.isEmpty else {
            errorMessage = "Please fill in all fields"
            return
        }
        
        isLoading = true

    guard let url = URL(string: "http://192.168.1.222:8000/auth/signup?username=\(username)&email=\(email)&password=\(password)") else {
            errorMessage = "Invalid URL"
            isLoading = false
            return
        }
        
        URLSession.shared.dataTask(with: url) { data, response, error in
            DispatchQueue.main.async {
                isLoading = false
                
                if let error = error {
                    errorMessage = "Network error: \(error.localizedDescription)"
                    return
                }
                
                guard let data = data else {
                    errorMessage = "No data received"
                    return
                }
                
                do {
                    if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                        if let status = json["status"] as? String {
                            if status == "success", let userId = json["user_id"] as? String {
                                currentUser = User(userId: userId, username: username, email: email)
                                isAuthenticated = true
                                showingSignup = false
                                clearForm()
                            } else {
                                errorMessage = json["message"] as? String ?? "Signup failed"
                            }
                        }
                    }
                } catch {
                    errorMessage = "Failed to parse response"
                }
            }
        }.resume()
    }
    
    func authenticateWithGoogle() {
        guard let userId = currentUser?.userId else {
            errorMessage = "User ID not found"
            return
        }
        
        isLoading = true

    guard let url = URL(string: "http://192.168.1.222:8000/auth/google?user_id=\(userId)") else {
            errorMessage = "Invalid URL"
            isLoading = false
            return
        }
        
        URLSession.shared.dataTask(with: url) { data, response, error in
            DispatchQueue.main.async {
                isLoading = false
                
                if let error = error {
                    errorMessage = "Network error: \(error.localizedDescription)"
                    return
                }
                
                guard let data = data else {
                    errorMessage = "No data received"
                    return
                }
                
                do {
                    if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                        print("Google auth response: \(json)")
                        if let status = json["status"] as? String {
                            if status == "auth_required", let authURLString = json["auth_url"] as? String {
                                authURL = authURLString
                                showingGoogleAuth = true
                            } else if status == "already_authenticated" {
                                // User already has valid credentials
                                googleAuthCompleted = true
                            } else {
                                errorMessage = json["message"] as? String ?? "Google auth failed"
                            }
                        } else {
                            errorMessage = "Invalid response format"
                        }
                    } else {
                        errorMessage = "Failed to parse JSON response"
                    }
                } catch {
                    errorMessage = "Failed to parse auth response: \(error.localizedDescription)"
                }
            }
        }.resume()
    }
    
    func completeGoogleAuth() {
        guard let userId = currentUser?.userId, !authCode.isEmpty else {
            errorMessage = "Missing user ID or authorization code"
            return
        }
        
        isLoading = true

    guard let url = URL(string: "http://192.168.1.222:8000/auth/google/complete") else {
            errorMessage = "Invalid URL"
            isLoading = false
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let requestBody = [
            "user_id": userId,
            "authorization_code": authCode.trimmingCharacters(in: .whitespacesAndNewlines)
        ]
        
        print("Sending OAuth completion request: \(requestBody)")
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            errorMessage = "Failed to encode request"
            isLoading = false
            return
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                isLoading = false
                
                if let error = error {
                    errorMessage = "Network error: \(error.localizedDescription)"
                    return
                }
                
                guard let data = data else {
                    errorMessage = "No data received"
                    return
                }
                
                do {
                    if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                        print("OAuth completion response: \(json)")
                        if let status = json["status"] as? String {
                            if status == "success" {
                                googleAuthCompleted = true
                                showingGoogleAuth = false
                                authCode = ""
                            } else {
                                errorMessage = json["message"] as? String ?? "Authentication failed"
                            }
                        } else {
                            errorMessage = "Invalid response format"
                        }
                    } else {
                        errorMessage = "Failed to parse JSON response"
                    }
                } catch {
                    errorMessage = "Failed to parse response: \(error.localizedDescription)"
                }
            }
        }.resume()
    }
    
    func clearForm() {
        username = ""
        email = ""
        password = ""
    }
}

// MARK: - Safari View for Google Auth
struct SafariView: UIViewControllerRepresentable {
    let url: URL
    
    func makeUIViewController(context: Context) -> SFSafariViewController {
        return SFSafariViewController(url: url)
    }
    
    func updateUIViewController(_ uiViewController: SFSafariViewController, context: Context) {}
}

#Preview {
    ContentView()
}

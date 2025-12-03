import Foundation
import SwiftUI

/// ViewModel for managing authentication UI state and logic
@MainActor
class AuthViewModel: ObservableObject {
    @Published var username = ""
    @Published var email = ""
    @Published var password = ""
    @Published var confirmPassword = ""
    
    @Published var selectedScopes: Set<OAuthScope> = [
        .widgetsRead,
        .widgetsWrite,
        .projectsRead,
        .projectsWrite,
        .filesRead
    ]
    
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var showError = false
    
    @Published var isLoginMode = true
    
    private let oauthService = OAuthService.shared
    
    var isFormValid: Bool {
        if isLoginMode {
            return !username.isEmpty && !password.isEmpty
        } else {
            return !username.isEmpty &&
                   !email.isEmpty &&
                   !password.isEmpty &&
                   password == confirmPassword &&
                   isValidEmail(email)
        }
    }
    
    var passwordsMatch: Bool {
        return password == confirmPassword || confirmPassword.isEmpty
    }
    
    private func isValidEmail(_ email: String) -> Bool {
        let emailRegex = "[A-Z0-9a-z._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,64}"
        let emailPredicate = NSPredicate(format: "SELF MATCHES %@", emailRegex)
        return emailPredicate.evaluate(with: email)
    }
    
    // MARK: - Actions
    
    func login() async {
        isLoading = true
        errorMessage = nil
        
        do {
            let scopeArray = Array(selectedScopes)
            _ = try await oauthService.login(
                username: username,
                password: password,
                scopes: scopeArray
            )
            
            // Success - OAuthService will update isAuthenticated
            clearForm()
        } catch let error as OAuthError {
            errorMessage = error.errorDescription
            showError = true
        } catch {
            errorMessage = "An unexpected error occurred"
            showError = true
        }
        
        isLoading = false
    }
    
    func signup() async {
        isLoading = true
        errorMessage = nil
        
        // Call signup endpoint
        guard let url = URL(string: "http://192.168.1.188:8000/auth/signup") else {
            errorMessage = "Invalid URL"
            showError = true
            isLoading = false
            return
        }
        
        var components = URLComponents(url: url, resolvingAgainstBaseURL: false)
        components?.queryItems = [
            URLQueryItem(name: "username", value: username),
            URLQueryItem(name: "email", value: email),
            URLQueryItem(name: "password", value: password)
        ]
        
        guard let finalUrl = components?.url else {
            errorMessage = "Failed to create request"
            showError = true
            isLoading = false
            return
        }
        
        var request = URLRequest(url: finalUrl)
        request.httpMethod = "POST"
        
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            
            guard let httpResponse = response as? HTTPURLResponse else {
                errorMessage = "Invalid response"
                showError = true
                isLoading = false
                return
            }
            
            if httpResponse.statusCode == 200 {
                // After successful signup, automatically login
                await login()
            } else {
                if let errorDict = try? JSONDecoder().decode([String: String].self, from: data),
                   let detail = errorDict["detail"] {
                    errorMessage = detail
                } else {
                    errorMessage = "Signup failed"
                }
                showError = true
            }
        } catch {
            errorMessage = "Network error: \(error.localizedDescription)"
            showError = true
        }
        
        isLoading = false
    }
    
    func toggleMode() {
        isLoginMode.toggle()
        clearForm()
    }
    
    private func clearForm() {
        username = ""
        email = ""
        password = ""
        confirmPassword = ""
        errorMessage = nil
    }
}

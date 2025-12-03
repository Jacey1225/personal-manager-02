import Foundation

/// Service for handling OAuth2 authentication flows
class OAuthService: ObservableObject {
    static let shared = OAuthService()
    
    private let baseURL = "http://192.168.1.188:8000"
    private let keychain = KeychainWrapper.shared
    
    @Published var isAuthenticated = false
    @Published var currentUser: AuthUser?
    @Published var grantedScopes: [String] = []
    
    private init() {
        // Check if user is already authenticated
        checkAuthState()
    }
    
    // MARK: - Authentication State
    
    func checkAuthState() {
        guard let accessToken = keychain.get(forKey: KeychainWrapper.accessTokenKey),
              let userId = keychain.get(forKey: KeychainWrapper.userIdKey) else {
            isAuthenticated = false
            currentUser = nil
            return
        }
        
        // Check if token is expired
        if isTokenExpired() {
            // Try to refresh
            Task {
                await refreshToken()
            }
        } else {
            isAuthenticated = true
            loadCurrentUser()
        }
    }
    
    private func loadCurrentUser() {
        guard let userId = keychain.get(forKey: KeychainWrapper.userIdKey) else { return }
        
        let scopesString = keychain.get(forKey: KeychainWrapper.grantedScopesKey) ?? ""
        let scopes = scopesString.split(separator: " ").map(String.init)
        
        currentUser = AuthUser(
            userId: userId,
            username: nil, // Could be loaded from API if needed
            email: nil,
            grantedScopes: scopes
        )
        grantedScopes = scopes
    }
    
    private func isTokenExpired() -> Bool {
        guard let expiryString = keychain.get(forKey: KeychainWrapper.tokenExpiryKey),
              let expiryTimestamp = Double(expiryString) else {
            return true
        }
        
        let expiryDate = Date(timeIntervalSince1970: expiryTimestamp)
        let fiveMinutesFromNow = Date().addingTimeInterval(5 * 60)
        
        return expiryDate <= fiveMinutesFromNow
    }
    
    // MARK: - Login
    
    func login(username: String, password: String, scopes: [OAuthScope]) async throws -> OAuthTokenResponse {
        guard let url = URL(string: "\(baseURL)/oauth/token") else {
            throw OAuthError.invalidURL
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        
        // Build form data
        let scopeString = scopes.map { $0.rawValue }.joined(separator: " ")
        let bodyParams = [
            "grant_type=password",
            "username=\(username.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? "")",
            "password=\(password.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? "")",
            "scope=\(scopeString.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? "")"
        ]
        request.httpBody = bodyParams.joined(separator: "&").data(using: .utf8)
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw OAuthError.invalidResponse
        }
        
        guard httpResponse.statusCode == 200 else {
            if let errorDict = try? JSONDecoder().decode([String: String].self, from: data),
               let detail = errorDict["detail"] {
                throw OAuthError.serverError(detail)
            }
            throw OAuthError.authenticationFailed
        }
        
        let tokenResponse = try JSONDecoder().decode(OAuthTokenResponse.self, from: data)
        
        // Store tokens securely
        saveTokens(tokenResponse)
        
        // Update state
        await MainActor.run {
            isAuthenticated = true
            loadCurrentUser()
        }
        
        return tokenResponse
    }
    
    // MARK: - Token Management
    
    private func saveTokens(_ response: OAuthTokenResponse) {
        _ = keychain.save(response.access_token, forKey: KeychainWrapper.accessTokenKey)
        _ = keychain.save(response.refresh_token, forKey: KeychainWrapper.refreshTokenKey)
        _ = keychain.save(String(response.expiryDate.timeIntervalSince1970), forKey: KeychainWrapper.tokenExpiryKey)
        _ = keychain.save(response.scope, forKey: KeychainWrapper.grantedScopesKey)
        
        if let userId = response.user_id {
            _ = keychain.save(userId, forKey: KeychainWrapper.userIdKey)
        }
    }
    
    func getValidToken() async -> String? {
        // Check if token is expired
        if isTokenExpired() {
            await refreshToken()
        }
        
        return keychain.get(forKey: KeychainWrapper.accessTokenKey)
    }
    
    @discardableResult
    func refreshToken() async -> Bool {
        guard let refreshToken = keychain.get(forKey: KeychainWrapper.refreshTokenKey),
              let url = URL(string: "\(baseURL)/oauth/token/refresh") else {
            await logout()
            return false
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let body = ["refresh_token": refreshToken]
        request.httpBody = try? JSONEncoder().encode(body)
        
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            
            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                await logout()
                return false
            }
            
            let tokenResponse = try JSONDecoder().decode(OAuthTokenResponse.self, from: data)
            saveTokens(tokenResponse)
            
            await MainActor.run {
                isAuthenticated = true
                loadCurrentUser()
            }
            
            return true
        } catch {
            await logout()
            return false
        }
    }
    
    // MARK: - Logout
    
    func logout() async {
        // Optionally revoke token on server
        if let token = keychain.get(forKey: KeychainWrapper.accessTokenKey),
           let url = URL(string: "\(baseURL)/oauth/revoke") {
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            
            let body = ["token": token]
            request.httpBody = try? JSONEncoder().encode(body)
            
            _ = try? await URLSession.shared.data(for: request)
        }
        
        // Clear keychain
        _ = keychain.clearAll()
        
        // Update state
        await MainActor.run {
            isAuthenticated = false
            currentUser = nil
            grantedScopes = []
        }
    }
}

// MARK: - Errors

enum OAuthError: LocalizedError {
    case invalidURL
    case invalidResponse
    case authenticationFailed
    case serverError(String)
    case networkError(Error)
    
    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid server URL"
        case .invalidResponse:
            return "Invalid response from server"
        case .authenticationFailed:
            return "Authentication failed. Please check your credentials."
        case .serverError(let message):
            return message
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        }
    }
}

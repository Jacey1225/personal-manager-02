import Foundation

/// Centralized API client with OAuth2 authentication
class APIClient {
    static let shared = APIClient()
    
    private let baseURL = "http://192.168.1.188:8000"
    private let oauthService = OAuthService.shared
    
    private init() {}
    
    // MARK: - Authenticated Requests
    
    /// Makes an authenticated GET request
    func get<T: Decodable>(
        _ endpoint: String,
        queryItems: [URLQueryItem]? = nil,
        responseType: T.Type
    ) async throws -> T {
        return try await request(
            endpoint: endpoint,
            method: "GET",
            queryItems: queryItems,
            body: nil,
            responseType: responseType
        )
    }
    
    /// Makes an authenticated POST request
    func post<T: Decodable, B: Encodable>(
        _ endpoint: String,
        body: B? = nil,
        queryItems: [URLQueryItem]? = nil,
        responseType: T.Type
    ) async throws -> T {
        return try await request(
            endpoint: endpoint,
            method: "POST",
            queryItems: queryItems,
            body: body,
            responseType: responseType
        )
    }
    
    /// Makes an authenticated DELETE request
    func delete<T: Decodable>(
        _ endpoint: String,
        queryItems: [URLQueryItem]? = nil,
        responseType: T.Type
    ) async throws -> T {
        return try await request(
            endpoint: endpoint,
            method: "DELETE",
            queryItems: queryItems,
            body: nil,
            responseType: responseType
        )
    }
    
    // MARK: - Private Request Handler
    
    private func request<T: Decodable, B: Encodable>(
        endpoint: String,
        method: String,
        queryItems: [URLQueryItem]?,
        body: B?,
        responseType: T.Type
    ) async throws -> T {
        // Get valid OAuth token (auto-refreshes if needed)
        guard let token = await oauthService.getValidToken() else {
            throw APIError.notAuthenticated
        }
        
        // Build URL
        guard var urlComponents = URLComponents(string: "\(baseURL)\(endpoint)") else {
            throw APIError.invalidURL
        }
        
        if let queryItems = queryItems {
            urlComponents.queryItems = queryItems
        }
        
        guard let url = urlComponents.url else {
            throw APIError.invalidURL
        }
        
        // Create request
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        // Add body if present
        if let body = body {
            request.httpBody = try JSONEncoder().encode(body)
        }
        
        // Execute request
        let (data, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }
        
        // Handle errors
        switch httpResponse.statusCode {
        case 200...299:
            // Success - decode response
            return try JSONDecoder().decode(T.self, from: data)
            
        case 401:
            // Unauthorized - token may be invalid
            throw APIError.unauthorized
            
        case 403:
            // Forbidden - insufficient scopes
            throw APIError.forbidden
            
        case 404:
            throw APIError.notFound
            
        default:
            // Try to decode error message
            if let errorDict = try? JSONDecoder().decode([String: String].self, from: data),
               let detail = errorDict["detail"] {
                throw APIError.serverError(detail)
            }
            throw APIError.serverError("Request failed with status \(httpResponse.statusCode)")
        }
    }
}

// MARK: - API Errors

enum APIError: LocalizedError {
    case invalidURL
    case invalidResponse
    case notAuthenticated
    case unauthorized
    case forbidden
    case notFound
    case serverError(String)
    
    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid URL"
        case .invalidResponse:
            return "Invalid response from server"
        case .notAuthenticated:
            return "Not authenticated. Please login."
        case .unauthorized:
            return "Unauthorized. Your session may have expired."
        case .forbidden:
            return "Forbidden. You don't have permission to access this resource."
        case .notFound:
            return "Resource not found"
        case .serverError(let message):
            return message
        }
    }
}

// MARK: - Usage Examples

extension APIClient {
    /// Example: Fetch user's projects
    func fetchProjects(userId: String) async throws -> [Project] {
        struct ProjectsResponse: Codable {
            let projects: [Project]
        }
        
        let response = try await get(
            "/projects/list",
            queryItems: [URLQueryItem(name: "user_id", value: userId)],
            responseType: ProjectsResponse.self
        )
        
        return response.projects
    }
    
    /// Example: Fetch project details
    func fetchProject(projectId: String, userId: String, projectName: String) async throws -> ProjectViewResponse {
        return try await get(
            "/projects/view_project",
            queryItems: [
                URLQueryItem(name: "project_id", value: projectId),
                URLQueryItem(name: "user_id", value: userId),
                URLQueryItem(name: "project_name", value: projectName),
                URLQueryItem(name: "force_refresh", value: "false")
            ],
            responseType: ProjectViewResponse.self
        )
    }
    
    /// Example: Fetch widgets for a project
    func fetchWidgets(projectId: String) async throws -> [String] {
        // This endpoint doesn't exist yet - example of how it would work
        struct WidgetsResponse: Codable {
            let widgets: [String]
        }
        
        let response = try await get(
            "/projects/\(projectId)/widgets",
            responseType: WidgetsResponse.self
        )
        
        return response.widgets
    }
}

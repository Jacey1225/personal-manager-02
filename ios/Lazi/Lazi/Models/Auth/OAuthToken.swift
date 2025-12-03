import Foundation

/// OAuth2 token response from the API
struct OAuthTokenResponse: Codable {
    let access_token: String
    let refresh_token: String
    let token_type: String
    let expires_in: Int
    let scope: String
    let user_id: String?
    
    var expiryDate: Date {
        return Date().addingTimeInterval(TimeInterval(expires_in))
    }
}

/// OAuth2 scope types
enum OAuthScope: String, CaseIterable, Identifiable {
    case widgetsRead = "widgets:read"
    case widgetsWrite = "widgets:write"
    case widgetsDelete = "widgets:delete"
    case widgetsAdmin = "widgets:admin"
    case projectsRead = "projects:read"
    case projectsWrite = "projects:write"
    case projectsDelete = "projects:delete"
    case projectsAdmin = "projects:admin"
    case filesRead = "files:read"
    case filesWrite = "files:write"
    case filesDelete = "files:delete"
    
    var id: String { rawValue }
    
    var displayName: String {
        switch self {
        case .widgetsRead: return "Read Widgets"
        case .widgetsWrite: return "Create/Edit Widgets"
        case .widgetsDelete: return "Delete Widgets"
        case .widgetsAdmin: return "Admin Widgets"
        case .projectsRead: return "Read Projects"
        case .projectsWrite: return "Create/Edit Projects"
        case .projectsDelete: return "Delete Projects"
        case .projectsAdmin: return "Admin Projects"
        case .filesRead: return "Read Files"
        case .filesWrite: return "Upload Files"
        case .filesDelete: return "Delete Files"
        }
    }
    
    var description: String {
        switch self {
        case .widgetsRead: return "View widget configurations and data"
        case .widgetsWrite: return "Create and modify widgets"
        case .widgetsDelete: return "Remove widgets from projects"
        case .widgetsAdmin: return "Full administrative control over widgets"
        case .projectsRead: return "View project details and members"
        case .projectsWrite: return "Create and edit projects"
        case .projectsDelete: return "Delete projects permanently"
        case .projectsAdmin: return "Full administrative control over projects"
        case .filesRead: return "Access uploaded files and media"
        case .filesWrite: return "Upload new files and media"
        case .filesDelete: return "Delete uploaded files"
        }
    }
    
    var category: String {
        if rawValue.starts(with: "widgets") { return "Widgets" }
        if rawValue.starts(with: "projects") { return "Projects" }
        if rawValue.starts(with: "files") { return "Files" }
        return "Other"
    }
}

/// User authentication state
struct AuthUser: Codable {
    let userId: String
    let username: String?
    let email: String?
    let grantedScopes: [String]
}

# OAuth Implementation - iOS

This document describes the OAuth2 authentication system implemented for the Lazi iOS app.

## Architecture Overview

The OAuth implementation is organized into modular, reusable components:

```
ios/Lazi/Lazi/
├── Services/
│   ├── KeychainWrapper.swift       # Secure token storage using iOS Keychain
│   └── OAuthService.swift          # OAuth2 flow logic & token management
├── Models/
│   └── Auth/
│       └── OAuthToken.swift        # Token response & scope enums
├── ViewModels/
│   └── AuthViewModel.swift         # Authentication UI state & validation
└── Views/
    └── Auth/
        ├── OAuthLoginView.swift    # Main login/signup screen
        └── Components/
            ├── AuthTextField.swift      # Reusable styled text field
            └── ScopeSelectionView.swift # OAuth scope selector
```

## Features

### ✅ OAuth2 Password Grant Flow
- Username/password authentication
- Configurable OAuth2 scopes
- Access token + refresh token management
- Automatic token refresh (5 minutes before expiry)
- Secure token storage in iOS Keychain

### ✅ User Registration
- Create new accounts via `/auth/signup`
- Auto-login after successful registration
- Email validation
- Password confirmation

### ✅ Scope Management
OAuth2 scopes are organized by category:

**Widgets:**
- `widgets:read` - View widget configurations
- `widgets:write` - Create/modify widgets
- `widgets:delete` - Delete widgets
- `widgets:admin` - Full widget admin access

**Projects:**
- `projects:read` - View projects
- `projects:write` - Create/modify projects
- `projects:delete` - Delete projects
- `projects:admin` - Full project admin access

**Files:**
- `files:read` - Access uploaded files
- `files:write` - Upload files
- `files:delete` - Delete files

### ✅ Security Features
- Secure Keychain storage for tokens
- Automatic token expiration handling
- Password visibility toggle
- Form validation
- Error handling with user-friendly messages

## Usage

### Basic Login Flow

```swift
import SwiftUI

struct MyView: View {
    @EnvironmentObject var oauthService: OAuthService
    
    var body: some View {
        if oauthService.isAuthenticated {
            // Show authenticated content
            Text("Welcome! User ID: \(oauthService.currentUser?.userId ?? "")")
            
            // Logout button
            Button("Logout") {
                Task {
                    await oauthService.logout()
                }
            }
        } else {
            // Show login view
            OAuthLoginView()
        }
    }
}
```

### Getting a Valid Access Token

The `OAuthService` automatically handles token refresh:

```swift
@EnvironmentObject var oauthService: OAuthService

func makeAuthenticatedRequest() async {
    guard let token = await oauthService.getValidToken() else {
        print("Not authenticated")
        return
    }
    
    var request = URLRequest(url: someURL)
    request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
    
    // Make request...
}
```

### Manual Login

```swift
let oauthService = OAuthService.shared

Task {
    do {
        let response = try await oauthService.login(
            username: "user@example.com",
            password: "password123",
            scopes: [
                .widgetsRead,
                .widgetsWrite,
                .projectsRead
            ]
        )
        
        print("Login successful! Token: \(response.access_token)")
        print("User ID: \(response.user_id ?? "")")
        print("Granted scopes: \(response.scope)")
    } catch let error as OAuthError {
        print("Login failed: \(error.errorDescription ?? "")")
    }
}
```

### Checking Granted Scopes

```swift
@EnvironmentObject var oauthService: OAuthService

var hasWidgetWriteAccess: Bool {
    oauthService.grantedScopes.contains("widgets:write")
}

// Or check multiple scopes
var canManageProjects: Bool {
    let requiredScopes = ["projects:read", "projects:write"]
    return requiredScopes.allSatisfy { oauthService.grantedScopes.contains($0) }
}
```

## API Endpoints

The OAuth service communicates with these backend endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/oauth/token` | POST | Login (password grant) |
| `/oauth/token/refresh` | POST | Refresh access token |
| `/oauth/revoke` | POST | Logout / revoke token |
| `/auth/signup` | POST | Create new account |

### Login Request Format

```
POST /oauth/token
Content-Type: application/x-www-form-urlencoded

grant_type=password&username=user@example.com&password=secret&scope=widgets:read widgets:write
```

### Token Response Format

```json
{
  "access_token": "eyJhbGci...",
  "refresh_token": "eyJhbGci...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "scope": "widgets:read widgets:write projects:read",
  "user_id": "usr_123abc"
}
```

## Token Lifecycle

1. **Login** → Access token (1 hour) + Refresh token (30 days)
2. **Token stored** → Securely saved in iOS Keychain
3. **API calls** → Auto-refresh if within 5 minutes of expiry
4. **Logout** → Tokens revoked on server + cleared from Keychain

## Component Isolation

Each component has a single responsibility:

- **KeychainWrapper** - Only handles secure storage
- **OAuthService** - Only handles OAuth flow & token management
- **AuthViewModel** - Only handles UI state & form validation
- **OAuthLoginView** - Only handles UI composition
- **AuthTextField** - Only handles styled text input
- **ScopeSelectionView** - Only handles scope selection

This makes the code:
- ✅ Easy to test
- ✅ Easy to modify
- ✅ Easy to reuse
- ✅ Easy to understand

## Testing

### Test Login with Existing User

1. Run the app
2. Enter username/password
3. Select desired scopes (or use defaults)
4. Tap "Sign In"

### Test Registration

1. Tap "Sign Up"
2. Enter username, email, password
3. Confirm password
4. Select scopes
5. Tap "Create Account"

### Test Token Refresh

The app automatically refreshes tokens. To test manually:

```swift
Task {
    let success = await OAuthService.shared.refreshToken()
    print("Refresh successful: \(success)")
}
```

### Test Logout

```swift
Task {
    await OAuthService.shared.logout()
    // App will automatically return to login screen
}
```

## Troubleshooting

### Login fails with "Invalid credentials"
- Check username/password are correct
- Verify backend is running on `http://192.168.1.188:8000`
- Check network connectivity

### Token refresh fails
- Refresh token may have expired (30 days)
- User must login again
- Check backend `/oauth/token/refresh` endpoint

### Scopes not working
- Check backend has scope validation enabled
- Verify correct scopes are requested during login
- Some endpoints require specific scopes

### Keychain access issues
- Keychain is sandboxed per app
- Uninstalling app clears Keychain
- Check device doesn't have Keychain disabled

## Future Enhancements

- [ ] Biometric authentication (Face ID / Touch ID)
- [ ] OAuth2 authorization code flow (for web apps)
- [ ] Social login (Google, Apple, GitHub)
- [ ] Remember me / persistent sessions
- [ ] Multi-factor authentication (MFA)
- [ ] Password reset flow
- [ ] Email verification
- [ ] Device management (trusted devices)

## Next Steps

Now that OAuth is implemented, you can:

1. ✅ Update existing API calls to use OAuth tokens
2. ✅ Add scope-based feature gating
3. ✅ Implement widget rendering (requires `widgets:read`)
4. ✅ Add widget creation UI (requires `widgets:write`)
5. ✅ Integrate with project management features

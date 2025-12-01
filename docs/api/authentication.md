# Authentication API Reference

OAuth2 authentication endpoints for the Lazi Public API.

## Overview

The Lazi API uses OAuth2 with JWT tokens for authentication. All API requests require a valid access token.

**Base Path**: `/public/oauth`

## Endpoints

### Request Access Token

Obtain an access token using username and password.

```
POST /oauth/token
```

#### Request

**Content-Type**: `application/x-www-form-urlencoded`

**Parameters**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `username` | string | Yes | User's username |
| `password` | string | Yes | User's password |
| `grant_type` | string | Yes | Must be "password" |
| `scope` | string | No | Space-separated list of scopes |

**Scopes**: See [OAuth2 Scopes](../advanced/scopes.md) for complete list.

#### Example Request

```bash
curl -X POST https://api.lazi.app/public/oauth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=john@example.com" \
  -d "password=secure_password_123" \
  -d "grant_type=password" \
  -d "scope=widgets:read widgets:write projects:read"
```

**Python Example**:

```python
import httpx

async def get_access_token():
    url = "https://api.lazi.app/public/oauth/token"
    
    data = {
        "username": "john@example.com",
        "password": "secure_password_123",
        "grant_type": "password",
        "scope": "widgets:read widgets:write projects:read"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, data=data)
        return response.json()
```

#### Response

**Status**: `200 OK`

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyXzEyM2FiYyIsInNjb3BlcyI6WyJ3aWRnZXRzOnJlYWQiLCJ3aWRnZXRzOndyaXRlIiwicHJvamVjdHM6cmVhZCJdLCJleHAiOjE3MDY3MDYwMDB9.signature",
  "refresh_token": "rt_abc123def456ghi789jkl012mno345pqr678",
  "token_type": "bearer",
  "expires_in": 3600,
  "scope": "widgets:read widgets:write projects:read",
  "user_id": "user_123abc"
}
```

**Response Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `access_token` | string | JWT access token (valid for 1 hour) |
| `refresh_token` | string | Refresh token (valid for 30 days) |
| `token_type` | string | Always "bearer" |
| `expires_in` | integer | Token expiration in seconds (3600) |
| `scope` | string | Granted scopes |
| `user_id` | string | User's unique identifier |

#### Error Responses

**401 Unauthorized** - Invalid credentials

```json
{
  "detail": "Invalid username or password",
  "error_code": "INVALID_CREDENTIALS"
}
```

**422 Unprocessable Entity** - Missing required fields

```json
{
  "detail": [
    {
      "loc": ["body", "username"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

---

### Refresh Token

Exchange a refresh token for a new access token.

```
POST /oauth/token/refresh
```

#### Request

**Content-Type**: `application/json`

**Body**:

```json
{
  "refresh_token": "rt_abc123def456ghi789jkl012mno345pqr678"
}
```

#### Example Request

```bash
curl -X POST https://api.lazi.app/public/oauth/token/refresh \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "rt_abc123def456ghi789jkl012mno345pqr678"
  }'
```

**Python Example**:

```python
import httpx

async def refresh_access_token(refresh_token: str):
    url = "https://api.lazi.app/public/oauth/token/refresh"
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            json={"refresh_token": refresh_token}
        )
        return response.json()
```

#### Response

**Status**: `200 OK`

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.new_token_payload.signature",
  "refresh_token": "rt_new123xyz456abc789def012ghi345jkl678",
  "token_type": "bearer",
  "expires_in": 3600,
  "scope": "widgets:read widgets:write projects:read",
  "user_id": "user_123abc"
}
```

#### Error Responses

**401 Unauthorized** - Invalid or expired refresh token

```json
{
  "detail": "Invalid or expired refresh token",
  "error_code": "INVALID_REFRESH_TOKEN"
}
```

---

### Revoke Token

Revoke an access or refresh token.

```
POST /oauth/revoke
```

#### Request

**Content-Type**: `application/json`

**Headers**:

```
Authorization: Bearer YOUR_ACCESS_TOKEN
```

**Body**:

```json
{
  "token": "rt_abc123def456ghi789jkl012mno345pqr678",
  "token_type_hint": "refresh_token"
}
```

**Fields**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `token` | string | Yes | Token to revoke |
| `token_type_hint` | string | No | "access_token" or "refresh_token" |

#### Example Request

```bash
curl -X POST https://api.lazi.app/public/oauth/revoke \
  -H "Authorization: Bearer eyJhbGci..." \
  -H "Content-Type: application/json" \
  -d '{
    "token": "rt_abc123...",
    "token_type_hint": "refresh_token"
  }'
```

**Python Example**:

```python
import httpx

async def revoke_token(access_token: str, refresh_token: str):
    url = "https://api.lazi.app/public/oauth/revoke"
    
    headers = {"Authorization": f"Bearer {access_token}"}
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            headers=headers,
            json={
                "token": refresh_token,
                "token_type_hint": "refresh_token"
            }
        )
        return response.status_code == 200
```

#### Response

**Status**: `200 OK`

```json
{
  "message": "Token revoked successfully"
}
```

#### Error Responses

**401 Unauthorized** - Missing or invalid authorization

```json
{
  "detail": "Not authenticated"
}
```

---

## Token Storage

### Best Practices

1. **Store Securely**: Never store tokens in:
   - Local storage (vulnerable to XSS)
   - URL parameters
   - Unencrypted files
   - Version control

2. **Use Secure Storage**:
   - Environment variables
   - Encrypted key stores
   - Secure credential managers
   - HTTP-only cookies (for web apps)

3. **Token Rotation**:
   - Refresh tokens before they expire
   - Implement automatic token refresh
   - Handle token refresh failures gracefully

### Example: Token Manager

```python
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

class TokenManager:
    """Secure token storage and management"""
    
    def __init__(self, token_file: str = ".lazi_tokens"):
        self.token_file = Path.home() / token_file
        self.tokens = self._load_tokens()
    
    def _load_tokens(self) -> dict:
        """Load tokens from encrypted file"""
        if not self.token_file.exists():
            return {}
        
        # In production, use encryption
        with open(self.token_file, 'r') as f:
            return json.load(f)
    
    def _save_tokens(self):
        """Save tokens to encrypted file"""
        # Set restrictive permissions
        self.token_file.touch(mode=0o600)
        
        with open(self.token_file, 'w') as f:
            json.dump(self.tokens, f)
    
    def store_tokens(self, access_token: str, refresh_token: str, 
                    expires_in: int):
        """Store tokens with expiration"""
        self.tokens = {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'expires_at': (
                datetime.utcnow() + timedelta(seconds=expires_in)
            ).isoformat()
        }
        self._save_tokens()
    
    def get_access_token(self) -> str | None:
        """Get valid access token"""
        if not self.tokens:
            return None
        
        expires_at = datetime.fromisoformat(
            self.tokens['expires_at']
        )
        
        if datetime.utcnow() >= expires_at:
            return None  # Token expired
        
        return self.tokens['access_token']
    
    def get_refresh_token(self) -> str | None:
        """Get refresh token"""
        return self.tokens.get('refresh_token')
    
    def clear_tokens(self):
        """Clear stored tokens"""
        self.tokens = {}
        if self.token_file.exists():
            self.token_file.unlink()
```

---

## OAuth2 Scopes

### Available Scopes

| Scope | Description |
|-------|-------------|
| `widgets:read` | Read widget data |
| `widgets:write` | Create and update widgets |
| `widgets:delete` | Delete widgets |
| `widgets:admin` | Full widget management |
| `projects:read` | Read project data |
| `projects:write` | Create and update projects |
| `projects:delete` | Delete projects |
| `projects:admin` | Full project management |
| `files:read` | Read file metadata |
| `files:write` | Upload files |
| `files:delete` | Delete files |

### Scope Hierarchies

Admin scopes include all lower permissions:

- `widgets:admin` → `widgets:delete` → `widgets:write` → `widgets:read`
- `projects:admin` → `projects:delete` → `projects:write` → `projects:read`
- `files:admin` → `files:delete` → `files:write` → `files:read`

### Requesting Scopes

Request minimum required scopes:

```python
# ✅ Good - minimal scopes
scopes = "widgets:read projects:read"

# ❌ Bad - excessive permissions
scopes = "widgets:admin projects:admin files:admin"
```

---

## Security Considerations

### Token Expiration

- **Access tokens**: 1 hour
- **Refresh tokens**: 30 days
- **Revoked tokens**: Immediately invalid

### Rate Limiting

Token endpoints have strict rate limits:

- **Token requests**: 10 per hour per IP
- **Token refresh**: 100 per hour per user
- **Token revoke**: 50 per hour per user

### IP Whitelisting

Configure IP whitelisting for enhanced security:

```python
# In your environment
ALLOWED_IPS = "192.168.1.0/24,10.0.0.0/8"
```

### Monitoring

Monitor token usage:

- Failed authentication attempts
- Token refresh patterns
- Unusual access patterns
- Token revocations

---

## Next Steps

- [Widget API](widgets.md) - Manage widgets
- [OAuth2 Scopes](../advanced/scopes.md) - Complete scope reference
- [Security Best Practices](../advanced/security.md) - Security guidelines

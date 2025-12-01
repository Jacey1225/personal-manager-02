# OAuth2 Scopes

Comprehensive guide to OAuth2 scopes in the Lazi API.

## Overview

OAuth2 scopes provide granular permission control for API access. Each scope represents a specific permission level for a resource type.

## Scope Format

Scopes follow the pattern: `resource:permission`

```
widgets:read      # Read widgets
projects:write    # Create/update projects
files:delete      # Delete files
```

## Available Scopes

### Widget Scopes

| Scope | Permission | Description |
|-------|-----------|-------------|
| `widgets:read` | Read | View widget data and metadata |
| `widgets:write` | Create/Update | Create new widgets and modify existing ones |
| `widgets:delete` | Delete | Delete widgets |
| `widgets:admin` | Full Control | All widget operations + manage permissions |

### Project Scopes

| Scope | Permission | Description |
|-------|-----------|-------------|
| `projects:read` | Read | View project details and members |
| `projects:write` | Create/Update | Create projects and modify settings |
| `projects:delete` | Delete | Delete projects |
| `projects:admin` | Full Control | All project operations + manage members |

### File Scopes

| Scope | Permission | Description |
|-------|-----------|-------------|
| `files:read` | Read | View file metadata and download files |
| `files:write` | Upload | Upload and attach media files |
| `files:delete` | Delete | Delete files |
| `files:admin` | Full Control | All file operations + manage storage |

### Organization Scopes

| Scope | Permission | Description |
|-------|-----------|-------------|
| `organizations:read` | Read | View organization details |
| `organizations:write` | Create/Update | Manage organization settings |
| `organizations:delete` | Delete | Delete organizations |
| `organizations:admin` | Full Control | All organization operations |

### User Scopes

| Scope | Permission | Description |
|-------|-----------|-------------|
| `user:read` | Read | View user profile |
| `user:write` | Update | Update user settings |
| `user:email` | Email Access | Access user email address |

### Admin Scopes

| Scope | Permission | Description |
|-------|-----------|-------------|
| `admin:users` | User Management | Manage all users |
| `admin:system` | System Control | System-level operations |

## Scope Hierarchies

Admin scopes include all lower-level permissions:

### Widget Hierarchy

```
widgets:admin
    ├── widgets:delete
    │   ├── widgets:write
    │   │   └── widgets:read
    │   └── widgets:read
    └── widgets:read
```

**Example**: A token with `widgets:admin` automatically has `widgets:delete`, `widgets:write`, and `widgets:read` permissions.

### Project Hierarchy

```
projects:admin
    ├── projects:delete
    │   ├── projects:write
    │   │   └── projects:read
    │   └── projects:read
    └── projects:read
```

### File Hierarchy

```
files:admin
    ├── files:delete
    │   ├── files:write
    │   │   └── files:read
    │   └── files:read
    └── files:read
```

## Requesting Scopes

### Minimum Required Scopes

Always request the **minimum scopes** needed for your application:

```python
# ✅ Good - minimal scopes
scopes = "widgets:read projects:read"

# ❌ Bad - excessive permissions
scopes = "widgets:admin projects:admin files:admin"
```

### OAuth2 Token Request

```python
import httpx

async def request_token_with_scopes():
    url = "https://api.lazi.app/public/oauth/token"
    
    data = {
        "username": "user@example.com",
        "password": "password",
        "grant_type": "password",
        "scope": "widgets:read widgets:write projects:read"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, data=data)
        token_data = response.json()
        
        # Verify granted scopes
        granted_scopes = token_data["scope"].split()
        print(f"Granted scopes: {granted_scopes}")
```

### Multiple Scopes

Request multiple scopes as a **space-separated** string:

```python
# Correct format
scope = "widgets:read widgets:write projects:read files:write"

# ❌ Wrong - comma-separated
scope = "widgets:read,widgets:write,projects:read"  # This will fail
```

## Scope Validation

### API Endpoint Requirements

Each endpoint specifies required scopes:

| Endpoint | HTTP Method | Required Scope |
|----------|------------|----------------|
| `GET /widgets` | GET | `widgets:read` |
| `POST /widgets` | POST | `widgets:write` |
| `PATCH /widgets/{id}` | PATCH | `widgets:write` |
| `DELETE /widgets/{id}` | DELETE | `widgets:delete` |
| `GET /projects` | GET | `projects:read` |
| `POST /projects` | POST | `projects:write` |
| `POST /widgets/{id}/media` | POST | `widgets:write` + `files:write` |

### Multiple Scope Requirements

Some endpoints require **multiple scopes**:

```python
# Attach media to widget requires both scopes
required_scopes = ["widgets:write", "files:write"]

# Your token must have both
token_scopes = "widgets:write files:write"
```

### Insufficient Scope Error

```json
{
  "detail": "Insufficient permissions. Required scope: widgets:write",
  "error_code": "INSUFFICIENT_SCOPE",
  "required_scopes": ["widgets:write"],
  "provided_scopes": ["widgets:read"]
}
```

## Scope Best Practices

### 1. Principle of Least Privilege

Grant only necessary permissions:

```python
# ✅ Good - read-only dashboard
scopes = "widgets:read projects:read"

# ✅ Good - widget creator app
scopes = "widgets:write files:write projects:read"

# ❌ Bad - requesting admin unnecessarily
scopes = "widgets:admin projects:admin"
```

### 2. Separate Service Accounts

Use different tokens for different purposes:

```python
# Analytics service (read-only)
analytics_scopes = "widgets:read projects:read"

# Widget creator service
creator_scopes = "widgets:write files:write"

# Admin dashboard
admin_scopes = "widgets:admin projects:admin"
```

### 3. Scope Rotation

Rotate scopes when requirements change:

```python
class APIClient:
    def __init__(self, token: str):
        self.token = token
        self.scopes = self._parse_token_scopes()
    
    def _parse_token_scopes(self) -> list[str]:
        """Extract scopes from JWT token"""
        import jwt
        payload = jwt.decode(self.token, options={"verify_signature": False})
        return payload.get("scopes", [])
    
    def can_perform(self, required_scope: str) -> bool:
        """Check if token has required scope"""
        return required_scope in self.scopes or \
               self._has_admin_scope(required_scope)
    
    def _has_admin_scope(self, required_scope: str) -> bool:
        """Check if admin scope covers required scope"""
        resource = required_scope.split(":")[0]
        return f"{resource}:admin" in self.scopes
```

### 4. Request Validation

Validate scopes before making requests:

```python
async def create_widget_safely(client: APIClient, widget_data: dict):
    if not client.can_perform("widgets:write"):
        raise PermissionError(
            "Token lacks widgets:write scope. "
            "Please request a new token with appropriate scopes."
        )
    
    # Proceed with widget creation
    return await client.create_widget(widget_data)
```

## Dynamic Scopes

### User-Specific Scopes

Scopes can be limited by user roles:

```python
# Regular user - limited scopes
user_scopes = "widgets:read widgets:write projects:read"

# Team admin - extended scopes
admin_scopes = "widgets:admin projects:write organizations:read"

# System admin - full access
system_scopes = "admin:users admin:system"
```

### Resource-Specific Scopes

Future enhancement: Scope to specific resources:

```python
# Access only specific project (planned feature)
scopes = "projects:proj_123abc:read widgets:proj_123abc:write"
```

## Scope Auditing

### Monitoring Scope Usage

Track which scopes are being used:

```python
import logging
from functools import wraps

def audit_scope_usage(required_scope: str):
    """Decorator to audit scope usage"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            logger = logging.getLogger("scope_audit")
            logger.info(f"Function {func.__name__} requires scope: {required_scope}")
            
            # Execute function
            result = await func(*args, **kwargs)
            
            logger.info(f"Scope {required_scope} used successfully")
            return result
        
        return wrapper
    return decorator

# Usage
@audit_scope_usage("widgets:write")
async def create_widget(data: dict):
    # Widget creation logic
    pass
```

### Scope Usage Analytics

```python
from collections import Counter
from datetime import datetime

class ScopeAnalytics:
    def __init__(self):
        self.usage_counter = Counter()
        self.last_used = {}
    
    def record_usage(self, scope: str):
        """Record scope usage"""
        self.usage_counter[scope] += 1
        self.last_used[scope] = datetime.utcnow()
    
    def get_most_used_scopes(self, top_n: int = 5):
        """Get most frequently used scopes"""
        return self.usage_counter.most_common(top_n)
    
    def get_unused_scopes(self, all_scopes: list[str]):
        """Find granted but unused scopes"""
        used_scopes = set(self.usage_counter.keys())
        return set(all_scopes) - used_scopes
```

## Scope Migration

### Adding New Scopes

When adding functionality, request additional scopes:

```python
async def upgrade_token_scopes(current_token: str, additional_scopes: str):
    """Request new token with additional scopes"""
    # Get current scopes from token
    import jwt
    payload = jwt.decode(current_token, options={"verify_signature": False})
    current_scopes = payload.get("scopes", [])
    
    # Combine with new scopes
    all_scopes = " ".join(current_scopes + additional_scopes.split())
    
    # Request new token
    # ... (token request logic)
```

### Deprecating Scopes

Handle deprecated scopes gracefully:

```python
DEPRECATED_SCOPES = {
    "widgets:full": "widgets:admin",  # Old -> New
    "projects:manage": "projects:write"
}

def normalize_scopes(scopes: list[str]) -> list[str]:
    """Convert deprecated scopes to new equivalents"""
    normalized = []
    
    for scope in scopes:
        if scope in DEPRECATED_SCOPES:
            new_scope = DEPRECATED_SCOPES[scope]
            print(f"Warning: Scope '{scope}' is deprecated. Use '{new_scope}' instead.")
            normalized.append(new_scope)
        else:
            normalized.append(scope)
    
    return list(set(normalized))  # Remove duplicates
```

## Security Considerations

### 1. Never Hardcode Scopes

```python
# ❌ Bad - hardcoded in source code
SCOPES = "widgets:admin projects:admin files:admin"

# ✅ Good - environment variable
SCOPES = os.getenv("API_SCOPES", "widgets:read projects:read")
```

### 2. Validate Scope Format

```python
import re

def validate_scope(scope: str) -> bool:
    """Validate scope format"""
    pattern = r'^[a-z]+:[a-z]+$'
    return bool(re.match(pattern, scope))

def validate_scopes(scopes: str) -> bool:
    """Validate multiple scopes"""
    for scope in scopes.split():
        if not validate_scope(scope):
            raise ValueError(f"Invalid scope format: {scope}")
    return True
```

### 3. Scope Injection Prevention

```python
def sanitize_scope_request(user_input: str) -> str:
    """Prevent scope injection attacks"""
    # Define allowed scopes
    ALLOWED_SCOPES = {
        "widgets:read", "widgets:write", "widgets:delete",
        "projects:read", "projects:write", "projects:delete",
        "files:read", "files:write", "files:delete"
    }
    
    # Parse and validate
    requested = set(user_input.split())
    valid = requested & ALLOWED_SCOPES
    
    if requested - valid:
        invalid = requested - valid
        raise ValueError(f"Invalid scopes requested: {invalid}")
    
    return " ".join(valid)
```

## Complete Example

```python
import os
import httpx
from typing import List

class ScopedAPIClient:
    """API client with scope management"""
    
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.token = None
        self.scopes = []
    
    async def authenticate(self, requested_scopes: str):
        """Authenticate with specific scopes"""
        url = "https://api.lazi.app/public/oauth/token"
        
        data = {
            "username": self.username,
            "password": self.password,
            "grant_type": "password",
            "scope": requested_scopes
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, data=data)
            
            if response.status_code == 200:
                token_data = response.json()
                self.token = token_data["access_token"]
                self.scopes = token_data["scope"].split()
                return True
            
            return False
    
    def requires_scope(self, scope: str):
        """Decorator to check scope before API call"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                if not self.has_scope(scope):
                    raise PermissionError(
                        f"Operation requires scope: {scope}. "
                        f"Available scopes: {', '.join(self.scopes)}"
                    )
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    def has_scope(self, required_scope: str) -> bool:
        """Check if client has required scope"""
        if required_scope in self.scopes:
            return True
        
        # Check for admin scope
        resource = required_scope.split(":")[0]
        return f"{resource}:admin" in self.scopes
    
    @requires_scope("widgets:read")
    async def list_widgets(self, project_id: str):
        """List widgets (requires widgets:read)"""
        # Implementation
        pass
    
    @requires_scope("widgets:write")
    async def create_widget(self, widget_data: dict):
        """Create widget (requires widgets:write)"""
        # Implementation
        pass

# Usage
async def main():
    client = ScopedAPIClient(
        username="user@example.com",
        password="password"
    )
    
    # Authenticate with minimal scopes
    await client.authenticate("widgets:read projects:read")
    
    # This works
    await client.list_widgets("proj_123")
    
    # This raises PermissionError
    try:
        await client.create_widget({"name": "New Widget"})
    except PermissionError as e:
        print(f"Error: {e}")
        
        # Re-authenticate with additional scopes
        await client.authenticate("widgets:write projects:read")
        
        # Now it works
        await client.create_widget({"name": "New Widget"})
```

## Next Steps

- [Authentication API](../api/authentication.md) - OAuth2 endpoints
- [Security Best Practices](security.md) - Security guidelines
- [Custom Endpoints](custom-endpoints.md) - Dynamic endpoints

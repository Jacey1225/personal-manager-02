# Security Best Practices

Comprehensive security guidelines for using the Lazi API.

## Overview

Security is paramount when building applications with the Lazi API. This guide covers best practices for authentication, data protection, code execution, and more.

## Authentication Security

### 1. Token Storage

**Never** store tokens insecurely:

```python
# ❌ NEVER DO THIS
# Hardcoded in source code
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

# In localStorage (vulnerable to XSS)
localStorage.setItem('token', token)

# In cookies without HttpOnly flag
document.cookie = `token=${token}`

# In version control
# config.py:
# API_TOKEN = "actual_token_here"
```

**✅ Secure token storage:**

```python
# Environment variables
import os
TOKEN = os.getenv("LAZI_API_TOKEN")

# Encrypted credential store
from keyring import get_password
TOKEN = get_password("lazi_api", "access_token")

# HTTP-only cookies (for web apps)
response.set_cookie(
    key="access_token",
    value=token,
    httponly=True,
    secure=True,  # HTTPS only
    samesite="strict"
)

# System keychain (macOS/Windows)
import keyring
keyring.set_password("lazi_api", "user@example.com", token)
```

### 2. Token Rotation

Implement automatic token refresh:

```python
import httpx
from datetime import datetime, timedelta

class TokenManager:
    def __init__(self):
        self.access_token = None
        self.refresh_token = None
        self.expires_at = None
    
    async def get_valid_token(self) -> str:
        """Get valid access token, refreshing if needed"""
        if not self.access_token or self._is_expired():
            await self.refresh()
        
        return self.access_token
    
    def _is_expired(self) -> bool:
        """Check if token is expired or about to expire"""
        if not self.expires_at:
            return True
        
        # Refresh 5 minutes before expiration
        buffer = timedelta(minutes=5)
        return datetime.utcnow() + buffer >= self.expires_at
    
    async def refresh(self):
        """Refresh access token"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.lazi.app/public/oauth/token/refresh",
                json={"refresh_token": self.refresh_token}
            )
            
            if response.status_code == 200:
                data = response.json()
                self.access_token = data["access_token"]
                self.refresh_token = data["refresh_token"]
                self.expires_at = datetime.utcnow() + timedelta(seconds=data["expires_in"])
            else:
                # Re-authenticate if refresh fails
                await self.authenticate()
```

### 3. Scope Minimization

Request only necessary scopes:

```python
# ✅ Good - minimal scopes
scopes = "widgets:read projects:read"

# ❌ Bad - excessive permissions
scopes = "widgets:admin projects:admin files:admin"

# ✅ Best - separate tokens for different operations
class APIClient:
    def __init__(self):
        # Read-only token for analytics
        self.read_token = self._get_token("widgets:read projects:read")
        
        # Write token for operations
        self.write_token = self._get_token("widgets:write files:write")
        
        # Admin token for management (separate service)
        self.admin_token = None  # Only when needed
```

### 4. Token Revocation

Revoke tokens when no longer needed:

```python
async def logout(token: str, refresh_token: str):
    """Revoke tokens on logout"""
    url = "https://api.lazi.app/public/oauth/revoke"
    
    headers = {"Authorization": f"Bearer {token}"}
    
    async with httpx.AsyncClient() as client:
        # Revoke refresh token
        await client.post(
            url,
            headers=headers,
            json={
                "token": refresh_token,
                "token_type_hint": "refresh_token"
            }
        )
        
        # Clear local storage
        delete_stored_tokens()
```

## API Security

### 1. HTTPS Only

**Always** use HTTPS:

```python
# ✅ Good
API_BASE_URL = "https://api.lazi.app"

# ❌ NEVER use HTTP in production
API_BASE_URL = "http://api.lazi.app"

# Enforce HTTPS in client
import httpx

class SecureClient:
    def __init__(self):
        # Timeout prevents hanging requests
        self.client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            verify=True  # Verify SSL certificates
        )
    
    async def request(self, method: str, url: str, **kwargs):
        if not url.startswith("https://"):
            raise ValueError("Only HTTPS URLs are allowed")
        
        return await self.client.request(method, url, **kwargs)
```

### 2. Input Validation

Validate all user input:

```python
from pydantic import BaseModel, validator, Field
import re

class WidgetCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(None, max_length=500)
    size: str = Field(default="small")
    
    @validator('name')
    def validate_name(cls, v):
        # No special characters that could cause issues
        if not re.match(r'^[a-zA-Z0-9\s\-_]+$', v):
            raise ValueError('Name contains invalid characters')
        return v
    
    @validator('size')
    def validate_size(cls, v):
        allowed = ['small', 'medium', 'large', 'extra_large']
        if v not in allowed:
            raise ValueError(f'Size must be one of: {", ".join(allowed)}')
        return v

# Usage
try:
    request = WidgetCreateRequest(
        name="Dashboard Widget",
        size="medium"
    )
except ValueError as e:
    print(f"Validation error: {e}")
```

### 3. Rate Limiting

Implement client-side rate limiting:

```python
import asyncio
from datetime import datetime, timedelta
from collections import deque

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window = timedelta(seconds=window_seconds)
        self.requests = deque()
    
    async def acquire(self):
        """Wait if rate limit would be exceeded"""
        now = datetime.utcnow()
        
        # Remove old requests outside window
        while self.requests and self.requests[0] < now - self.window:
            self.requests.popleft()
        
        # Check if at limit
        if len(self.requests) >= self.max_requests:
            # Calculate wait time
            oldest = self.requests[0]
            wait_until = oldest + self.window
            wait_seconds = (wait_until - now).total_seconds()
            
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
                return await self.acquire()
        
        # Record this request
        self.requests.append(now)

# Usage
limiter = RateLimiter(max_requests=100, window_seconds=3600)

async def create_widget_with_rate_limit(data: dict):
    await limiter.acquire()
    # Now safe to make request
    return await create_widget(data)
```

### 4. Retry Logic with Exponential Backoff

Handle transient failures securely:

```python
import asyncio
import random

async def retry_with_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0
):
    """Retry function with exponential backoff"""
    
    for attempt in range(max_retries):
        try:
            return await func()
        
        except httpx.HTTPStatusError as e:
            # Don't retry client errors (4xx)
            if 400 <= e.response.status_code < 500:
                raise
            
            # Retry server errors (5xx) and timeouts
            if attempt < max_retries - 1:
                # Exponential backoff with jitter
                delay = min(
                    base_delay * (2 ** attempt) + random.uniform(0, 1),
                    max_delay
                )
                
                await asyncio.sleep(delay)
            else:
                raise
        
        except httpx.TimeoutException:
            if attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), max_delay)
                await asyncio.sleep(delay)
            else:
                raise

# Usage
async def create_widget_resilient(data: dict):
    return await retry_with_backoff(
        lambda: create_widget(data),
        max_retries=3
    )
```

## Code Execution Security

### 1. Never Execute Untrusted Code Directly

```python
# ❌ EXTREMELY DANGEROUS
user_code = request.json.get('code')
exec(user_code)  # Never do this!

# ✅ Use sandboxed execution
from api.config.strict_pyenv import validate_code, ValidateRunTime

def execute_user_code_safely(code: str, context: dict):
    # 1. Validate code
    forbidden = validate_code(code)
    if forbidden:
        raise ValueError(f"Forbidden modules: {forbidden}")
    
    # 2. Execute in sandbox
    validator = ValidateRunTime(timeout=5.0, memory_limit=1024.0)
    result = validator.run(code, context)
    
    return result
```

### 2. Restrict Imports

```python
# Define allowed modules
ALLOWED_MODULES = {
    'datetime', 'time', 'json', 're', 'math',
    'statistics', 'collections', 'itertools'
}

def check_imports(code: str) -> bool:
    """Check if code only uses allowed modules"""
    import ast
    
    tree = ast.parse(code)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split('.')[0]
                if module not in ALLOWED_MODULES:
                    raise ValueError(f"Module '{module}' not allowed")
        
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module = node.module.split('.')[0]
                if module not in ALLOWED_MODULES:
                    raise ValueError(f"Module '{module}' not allowed")
    
    return True
```

### 3. Timeout Protection

```python
import signal
from contextlib import contextmanager

@contextmanager
def timeout(seconds: int):
    """Context manager for timeouts"""
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation exceeded {seconds} seconds")
    
    # Set alarm
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Cancel alarm
        signal.alarm(0)

# Usage
try:
    with timeout(5):
        result = execute_user_code(code)
except TimeoutError as e:
    print(f"Code execution timed out: {e}")
```

## Data Protection

### 1. Sensitive Data Handling

```python
import os
from cryptography.fernet import Fernet

class SecureConfig:
    """Encrypted configuration storage"""
    
    def __init__(self):
        # Get encryption key from environment
        key = os.getenv('ENCRYPTION_KEY')
        if not key:
            raise ValueError("ENCRYPTION_KEY not set")
        
        self.cipher = Fernet(key.encode())
    
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher.decrypt(encrypted.encode()).decode()

# Usage
config = SecureConfig()

# Encrypt API keys before storage
encrypted_key = config.encrypt(api_key)
save_to_database(encrypted_key)

# Decrypt when needed
api_key = config.decrypt(encrypted_key)
```

### 2. Logging Security

```python
import logging
import re

class SecureFormatter(logging.Formatter):
    """Formatter that redacts sensitive information"""
    
    PATTERNS = [
        (r'Bearer\s+[A-Za-z0-9\-_=]+\.[A-Za-z0-9\-_=]+\.[A-Za-z0-9\-_=]+', 'Bearer [REDACTED]'),
        (r'"password"\s*:\s*"[^"]*"', '"password": "[REDACTED]"'),
        (r'"token"\s*:\s*"[^"]*"', '"token": "[REDACTED]"'),
        (r'api_key=[A-Za-z0-9]+', 'api_key=[REDACTED]'),
    ]
    
    def format(self, record):
        message = super().format(record)
        
        # Redact sensitive patterns
        for pattern, replacement in self.PATTERNS:
            message = re.sub(pattern, replacement, message)
        
        return message

# Configure logging
handler = logging.StreamHandler()
handler.setFormatter(SecureFormatter())

logger = logging.getLogger('api')
logger.addHandler(handler)

# Safe logging
logger.info(f"Request: Authorization: Bearer {token}")
# Output: "Request: Authorization: Bearer [REDACTED]"
```

### 3. Database Security

```python
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.encryption_options import AutoEncryptionOpts

class SecureMongoClient:
    """MongoDB client with encryption"""
    
    def __init__(self):
        # Client-side field level encryption
        kms_providers = {
            "local": {
                "key": os.getenv("MONGO_ENCRYPTION_KEY").encode()
            }
        }
        
        auto_encryption_opts = AutoEncryptionOpts(
            kms_providers=kms_providers,
            key_vault_namespace="encryption.__keyVault"
        )
        
        self.client = AsyncIOMotorClient(
            os.getenv("MONGO_URI"),
            auto_encryption_opts=auto_encryption_opts
        )
    
    async def store_sensitive_data(self, collection: str, data: dict):
        """Store data with automatic encryption"""
        db = self.client[os.getenv("DB_NAME")]
        return await db[collection].insert_one(data)
```

## Network Security

### 1. Certificate Validation

```python
import httpx
import certifi

# Always verify SSL certificates
client = httpx.AsyncClient(verify=certifi.where())

# Don't disable verification in production!
# client = httpx.AsyncClient(verify=False)  # ❌ NEVER DO THIS
```

### 2. Request Signing

```python
import hmac
import hashlib
import time

def sign_request(payload: str, secret: str) -> str:
    """Sign request payload"""
    timestamp = str(int(time.time()))
    message = f"{timestamp}.{payload}"
    
    signature = hmac.new(
        secret.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return f"{timestamp}.{signature}"

def verify_signature(payload: str, signature: str, secret: str) -> bool:
    """Verify request signature"""
    try:
        timestamp, sig = signature.split('.')
        
        # Check timestamp (prevent replay attacks)
        if abs(int(time.time()) - int(timestamp)) > 300:  # 5 minutes
            return False
        
        # Verify signature
        expected = sign_request(payload, secret).split('.')[1]
        return hmac.compare_digest(sig, expected)
    
    except Exception:
        return False
```

### 3. IP Whitelisting

```python
from fastapi import Request, HTTPException
import ipaddress

ALLOWED_IP_RANGES = [
    ipaddress.ip_network('192.168.1.0/24'),
    ipaddress.ip_network('10.0.0.0/8'),
]

async def check_ip_whitelist(request: Request):
    """Middleware to check IP whitelist"""
    client_ip = ipaddress.ip_address(request.client.host)
    
    allowed = any(
        client_ip in network
        for network in ALLOWED_IP_RANGES
    )
    
    if not allowed:
        raise HTTPException(
            status_code=403,
            detail="IP address not allowed"
        )
```

## Monitoring and Auditing

### 1. Security Event Logging

```python
import logging
from datetime import datetime

class SecurityLogger:
    """Log security-relevant events"""
    
    def __init__(self):
        self.logger = logging.getLogger('security')
    
    def log_authentication(self, user_id: str, success: bool, ip: str):
        """Log authentication attempt"""
        self.logger.info({
            'event': 'authentication',
            'user_id': user_id,
            'success': success,
            'ip': ip,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def log_permission_denied(self, user_id: str, resource: str, action: str):
        """Log permission denial"""
        self.logger.warning({
            'event': 'permission_denied',
            'user_id': user_id,
            'resource': resource,
            'action': action,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def log_suspicious_activity(self, user_id: str, description: str):
        """Log suspicious activity"""
        self.logger.error({
            'event': 'suspicious_activity',
            'user_id': user_id,
            'description': description,
            'timestamp': datetime.utcnow().isoformat()
        })
```

### 2. Failed Login Detection

```python
from collections import defaultdict
from datetime import datetime, timedelta

class LoginMonitor:
    """Detect brute force attacks"""
    
    def __init__(self, max_attempts: int = 5, window_minutes: int = 15):
        self.max_attempts = max_attempts
        self.window = timedelta(minutes=window_minutes)
        self.attempts = defaultdict(list)
    
    def record_failed_login(self, username: str, ip: str):
        """Record failed login attempt"""
        key = f"{username}:{ip}"
        self.attempts[key].append(datetime.utcnow())
        
        # Clean old attempts
        cutoff = datetime.utcnow() - self.window
        self.attempts[key] = [
            t for t in self.attempts[key]
            if t > cutoff
        ]
        
        # Check if locked out
        if len(self.attempts[key]) >= self.max_attempts:
            return True  # Account locked
        
        return False
    
    def clear_attempts(self, username: str, ip: str):
        """Clear attempts after successful login"""
        key = f"{username}:{ip}"
        if key in self.attempts:
            del self.attempts[key]
```

## Compliance

### 1. GDPR Considerations

```python
class GDPRCompliance:
    """GDPR compliance utilities"""
    
    async def export_user_data(self, user_id: str) -> dict:
        """Export all user data (GDPR right to data portability)"""
        return {
            'user': await self.get_user(user_id),
            'widgets': await self.get_user_widgets(user_id),
            'projects': await self.get_user_projects(user_id),
            'activity_log': await self.get_user_activity(user_id)
        }
    
    async def delete_user_data(self, user_id: str):
        """Delete all user data (GDPR right to erasure)"""
        # Delete user data
        await self.delete_user(user_id)
        await self.delete_user_widgets(user_id)
        await self.delete_user_projects(user_id)
        
        # Anonymize logs (keep for legal reasons)
        await self.anonymize_user_logs(user_id)
```

### 2. Data Retention

```python
from datetime import datetime, timedelta

class DataRetentionPolicy:
    """Implement data retention policies"""
    
    RETENTION_PERIODS = {
        'access_logs': timedelta(days=90),
        'audit_logs': timedelta(days=365),
        'user_data': None,  # Keep until deleted
    }
    
    async def cleanup_old_data(self):
        """Remove data past retention period"""
        now = datetime.utcnow()
        
        for data_type, retention in self.RETENTION_PERIODS.items():
            if retention:
                cutoff = now - retention
                await self.delete_data_before(data_type, cutoff)
```

## Security Checklist

Use this checklist before deploying:

- [ ] All tokens stored securely (environment variables, encrypted storage)
- [ ] HTTPS enforced for all API requests
- [ ] SSL certificate verification enabled
- [ ] Input validation on all user inputs
- [ ] Rate limiting implemented
- [ ] Retry logic with exponential backoff
- [ ] User code executed in sandbox
- [ ] Forbidden modules blocked
- [ ] Execution timeouts configured
- [ ] Sensitive data encrypted at rest
- [ ] Logs redact sensitive information
- [ ] Failed login detection active
- [ ] Security events logged
- [ ] IP whitelisting configured (if applicable)
- [ ] GDPR compliance measures implemented
- [ ] Data retention policies configured
- [ ] Regular security audits scheduled

## Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [OAuth 2.0 Security Best Practices](https://tools.ietf.org/html/draft-ietf-oauth-security-topics)
- [Python Security Best Practices](https://python.readthedocs.io/en/latest/library/security_warnings.html)

## Next Steps

- [OAuth2 Scopes](scopes.md) - Scope management
- [Custom Endpoints](custom-endpoints.md) - Dynamic endpoints
- [Authentication API](../api/authentication.md) - Authentication endpoints

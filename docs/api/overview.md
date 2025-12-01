# API Reference Overview

Complete reference for the Lazi Public API.

## Base URL

```
https://api.lazi.app/public
```

For development:
```
http://localhost:8000/public
```

## Authentication

All API endpoints require OAuth2 authentication. Include your access token in the `Authorization` header:

```bash
Authorization: Bearer YOUR_ACCESS_TOKEN
```

See [Authentication Reference](authentication.md) for detailed endpoint documentation.

## API Versioning

The API uses URL-based versioning:

- **Current Version**: v1 (implicit, no version prefix required)
- **Future Versions**: Will be available at `/public/v2`, etc.

## Content Types

### Request Content Types

- `application/json` - For JSON request bodies
- `multipart/form-data` - For file uploads
- `application/x-www-form-urlencoded` - For OAuth2 token requests

### Response Content Types

All API responses return `application/json` unless otherwise specified.

## Response Format

### Success Response

```json
{
  "success": true,
  "data": {
    // Response data
  },
  "metadata": {
    "timestamp": "2024-01-31T12:00:00Z",
    "request_id": "req_123abc"
  }
}
```

### Error Response

```json
{
  "detail": "Error message description",
  "error_code": "INVALID_TOKEN",
  "status_code": 401
}
```

## Common HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| `200` | OK | Request succeeded |
| `201` | Created | Resource created successfully |
| `204` | No Content | Request succeeded, no content returned |
| `400` | Bad Request | Invalid request parameters |
| `401` | Unauthorized | Missing or invalid authentication |
| `403` | Forbidden | Insufficient permissions |
| `404` | Not Found | Resource doesn't exist |
| `409` | Conflict | Resource conflict (e.g., duplicate) |
| `422` | Unprocessable Entity | Validation error |
| `429` | Too Many Requests | Rate limit exceeded |
| `500` | Internal Server Error | Server error |
| `503` | Service Unavailable | Temporary outage |

## Rate Limiting

### Limits

- **Standard**: 1000 requests per hour
- **Widgets:write**: 100 widget creations per hour
- **Files**: 50 file uploads per hour

### Rate Limit Headers

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 987
X-RateLimit-Reset: 1706702400
```

### Rate Limit Exceeded Response

```json
{
  "detail": "Rate limit exceeded",
  "error_code": "RATE_LIMIT_EXCEEDED",
  "status_code": 429,
  "retry_after": 3600
}
```

## Pagination

For endpoints that return lists, pagination is available:

### Request Parameters

```
?page=1&per_page=50
```

- `page` - Page number (default: 1)
- `per_page` - Items per page (default: 20, max: 100)

### Response Format

```json
{
  "data": [...],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total_items": 156,
    "total_pages": 8,
    "has_next": true,
    "has_previous": false
  }
}
```

## Filtering and Sorting

### Filtering

Use query parameters to filter results:

```
GET /widgets?project_id=proj_123&size=medium
```

### Sorting

Sort results using the `sort` parameter:

```
GET /widgets?sort=created_at:desc
```

- Format: `field:order`
- Order: `asc` (ascending) or `desc` (descending)
- Multiple sorts: `sort=created_at:desc,name:asc`

## Timestamps

All timestamps use ISO 8601 format with UTC timezone:

```
2024-01-31T12:00:00.000Z
```

## Idempotency

For safe retry of requests, use the `Idempotency-Key` header:

```bash
POST /widgets
Idempotency-Key: unique-request-id-12345
```

The same `Idempotency-Key` within 24 hours will return the cached response instead of creating a duplicate resource.

## CORS

Cross-Origin Resource Sharing (CORS) is enabled for all origins in development. In production, configure allowed origins via environment variables.

### CORS Headers

```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, PUT, PATCH, DELETE, OPTIONS
Access-Control-Allow-Headers: Authorization, Content-Type, Idempotency-Key
Access-Control-Max-Age: 3600
```

## Webhooks

Configure webhooks to receive real-time notifications:

### Events

- `widget.created`
- `widget.updated`
- `widget.deleted`
- `widget.interaction`
- `project.created`
- `project.updated`

### Webhook Payload

```json
{
  "event": "widget.created",
  "timestamp": "2024-01-31T12:00:00Z",
  "data": {
    "widget_id": "widget_123abc",
    "project_id": "proj_456def",
    "user_id": "user_789ghi"
  }
}
```

### Webhook Signature

Verify webhook authenticity using the `X-Lazi-Signature` header:

```python
import hmac
import hashlib

def verify_webhook(payload, signature, secret):
    expected = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)
```

## SDK Support

Official SDKs are available for:

- **Python**: `pip install lazi-sdk`
- **JavaScript/TypeScript**: `npm install @lazi/sdk` (coming soon)
- **Go**: `go get github.com/lazi/go-sdk` (coming soon)

## API Endpoints

### Authentication

- [POST /oauth/token](authentication.md#request-access-token) - Request access token
- [POST /oauth/token/refresh](authentication.md#refresh-token) - Refresh access token
- [POST /oauth/revoke](authentication.md#revoke-token) - Revoke token

### Widgets

- [GET /widgets](widgets.md#list-widgets) - List widgets
- [GET /widgets/{id}](widgets.md#get-widget) - Get widget details
- [POST /widgets](widgets.md#create-widget) - Create widget
- [PATCH /widgets/{id}](widgets.md#update-widget) - Update widget
- [DELETE /widgets/{id}](widgets.md#delete-widget) - Delete widget
- [POST /widgets/{id}/media](widgets.md#attach-media) - Attach media
- [POST /widgets/{id}/interact](widgets.md#record-interaction) - Record interaction

### Projects

- [GET /projects](projects.md#list-projects) - List projects
- [GET /projects/{id}](projects.md#get-project) - Get project details
- [POST /projects](projects.md#create-project) - Create project
- [PATCH /projects/{id}](projects.md#update-project) - Update project
- [DELETE /projects/{id}](projects.md#delete-project) - Delete project
- [GET /projects/{id}/widgets](projects.md#list-project-widgets) - List project widgets

## Example: Complete Request

```bash
curl -X POST https://api.lazi.app/public/widgets \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: req-123abc-456def" \
  -d '{
    "name": "Sales Dashboard",
    "description": "Real-time sales metrics",
    "size": "large",
    "project_id": "proj_123abc",
    "endpoint": "https://api.example.com/sales",
    "endpoint_data": {
      "refresh_interval": 300,
      "logic": "def process_data(data, req, user): return data"
    }
  }'
```

## Support

- **Documentation**: [https://docs.lazi.app](https://docs.lazi.app)
- **GitHub**: [https://github.com/Jacey1225/personal-manager-02](https://github.com/Jacey1225/personal-manager-02)
- **Issues**: [GitHub Issues](https://github.com/Jacey1225/personal-manager-02/issues)

## Changelog

Track API changes and deprecations:

- **2024-01-31**: Initial public API release (v1)
- OAuth2 with scopes
- Widget management endpoints
- Project management endpoints
- File upload support

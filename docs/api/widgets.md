# Widgets API Reference

Widget management endpoints for the Lazi Public API.

## Overview

Widgets are customizable dashboard components that can display data from external sources, execute custom logic, and include media attachments.

**Base Path**: `/public/widgets`

**Required Scopes**: 
- `widgets:read` - Read widget data
- `widgets:write` - Create/update widgets
- `widgets:delete` - Delete widgets

## Endpoints

### List Widgets

Retrieve a list of widgets for a specific project.

```
GET /widgets
```

#### Request

**Headers**:

```
Authorization: Bearer YOUR_ACCESS_TOKEN
```

**Query Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `project_id` | string | Yes | Project identifier |
| `page` | integer | No | Page number (default: 1) |
| `per_page` | integer | No | Items per page (default: 20, max: 100) |
| `size` | string | No | Filter by size: "small", "medium", "large", "extra_large" |
| `sort` | string | No | Sort field and order (e.g., "created_at:desc") |

#### Example Request

```bash
curl -X GET "https://api.lazi.app/public/widgets?project_id=proj_123abc&per_page=10" \
  -H "Authorization: Bearer eyJhbGciOi..."
```

**Python Example**:

```python
from lazi_sdk import ReadWidget

async def list_widgets():
    reader = ReadWidget(
        username="john@example.com",
        token="eyJhbGciOi...",
        project_id="proj_123abc"
    )
    
    widgets = await reader.list_widgets()
    
    for widget in widgets:
        print(f"{widget.name} - {widget.size}")
```

#### Response

**Status**: `200 OK`

```json
{
  "data": [
    {
      "id": "widget_123abc",
      "name": "Sales Dashboard",
      "description": "Real-time sales metrics",
      "content": [
        "https://s3.amazonaws.com/bucket/presigned-url-1"
      ],
      "size": "large",
      "interaction": {
        "https://api.example.com/sales": {
          "params": {"period": "weekly"},
          "headers": {"Authorization": "Bearer token"},
          "refresh_interval": 300,
          "logic": "def process_data(data, req, user): return data"
        }
      },
      "created_at": "2024-01-31T12:00:00Z",
      "updated_at": "2024-01-31T12:00:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 10,
    "total_items": 23,
    "total_pages": 3,
    "has_next": true,
    "has_previous": false
  }
}
```

---

### Get Widget

Retrieve details of a specific widget.

```
GET /widgets/{widget_id}
```

#### Request

**Headers**:

```
Authorization: Bearer YOUR_ACCESS_TOKEN
```

**Path Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `widget_id` | string | Yes | Widget identifier |

#### Example Request

```bash
curl -X GET "https://api.lazi.app/public/widgets/widget_123abc" \
  -H "Authorization: Bearer eyJhbGciOi..."
```

**Python Example**:

```python
from lazi_sdk import ReadWidget

async def get_widget_details():
    reader = ReadWidget(
        username="john@example.com",
        token="eyJhbGciOi...",
        project_id="proj_123abc"
    )
    
    widget = await reader.get_widget("widget_123abc")
    print(f"Widget: {widget.name}")
    print(f"Size: {widget.size}")
    print(f"Media files: {len(widget.content)}")
```

#### Response

**Status**: `200 OK`

```json
{
  "id": "widget_123abc",
  "name": "Sales Dashboard",
  "description": "Real-time sales metrics",
  "content": [
    "https://s3.amazonaws.com/bucket/presigned-url-1"
  ],
  "size": "large",
  "interaction": {
    "https://api.example.com/sales": {
      "params": {"period": "weekly"},
      "headers": {"Authorization": "Bearer token"},
      "refresh_interval": 300,
      "logic": "def process_data(data, req, user): return data"
    }
  },
  "project_id": "proj_123abc",
  "user_id": "user_789xyz",
  "created_at": "2024-01-31T12:00:00Z",
  "updated_at": "2024-01-31T12:00:00Z"
}
```

---

### Create Widget

Create a new widget.

```
POST /widgets
```

#### Request

**Headers**:

```
Authorization: Bearer YOUR_ACCESS_TOKEN
Content-Type: application/json
```

**Required Scope**: `widgets:write`

**Body**:

```json
{
  "name": "Sales Dashboard",
  "description": "Real-time sales metrics",
  "size": "large",
  "project_id": "proj_123abc",
  "endpoint": "https://api.example.com/sales",
  "endpoint_data": {
    "params": {
      "period": "weekly",
      "metrics": "revenue,users"
    },
    "headers": {
      "Authorization": "Bearer external_token"
    },
    "refresh_interval": 300,
    "logic": "def process_data(widget_data, request, user):\n    return {'revenue': widget_data.get('revenue', 0)}"
  }
}
```

**Request Schema**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Widget name |
| `description` | string | No | Widget description |
| `size` | string | No | Widget size: "small", "medium", "large", "extra_large" (default: "small") |
| `project_id` | string | Yes | Project identifier |
| `endpoint` | string | No | External API endpoint |
| `endpoint_data` | object | No | Endpoint configuration |
| `endpoint_data.params` | object | No | Query parameters for endpoint |
| `endpoint_data.headers` | object | No | HTTP headers for endpoint |
| `endpoint_data.refresh_interval` | integer | No | Refresh interval in seconds (0 = no refresh) |
| `endpoint_data.logic` | string | No | Python function to process endpoint data |

#### Example Request

```bash
curl -X POST "https://api.lazi.app/public/widgets" \
  -H "Authorization: Bearer eyJhbGciOi..." \
  -H "Content-Type: application/json" \
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

**Python Example**:

```python
from lazi_sdk import WriteWidget

async def create_sales_widget():
    widget = WriteWidget(
        token="eyJhbGciOi...",
        user_id="user_789xyz"
    )
    
    await widget.create_new(
        name="Sales Dashboard",
        description="Real-time sales metrics",
        size="large",
        project_id="proj_123abc"
    )
    
    await widget.post(
        endpoint="https://api.example.com/sales",
        endpoint_data={
            "params": {"period": "weekly"},
            "refresh_interval": 300,
            "logic": "def process_data(data, req, user): return data"
        }
    )
    
    await widget.save()
    print(f"Widget created: {widget.current_widget.id}")
```

#### Response

**Status**: `201 Created`

```json
{
  "id": "widget_123abc",
  "name": "Sales Dashboard",
  "description": "Real-time sales metrics",
  "content": [],
  "size": "large",
  "interaction": {
    "https://api.example.com/sales": {
      "params": {"period": "weekly"},
      "headers": {},
      "refresh_interval": 300,
      "logic": "def process_data(data, req, user): return data"
    }
  },
  "project_id": "proj_123abc",
  "user_id": "user_789xyz",
  "created_at": "2024-01-31T12:00:00Z",
  "updated_at": "2024-01-31T12:00:00Z"
}
```

---

### Update Widget

Update an existing widget.

```
PATCH /widgets/{widget_id}
```

#### Request

**Headers**:

```
Authorization: Bearer YOUR_ACCESS_TOKEN
Content-Type: application/json
```

**Required Scope**: `widgets:write`

**Path Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `widget_id` | string | Yes | Widget identifier |

**Body**: (All fields optional, include only fields to update)

```json
{
  "name": "Updated Dashboard Name",
  "description": "Updated description",
  "size": "extra_large"
}
```

#### Example Request

```bash
curl -X PATCH "https://api.lazi.app/public/widgets/widget_123abc" \
  -H "Authorization: Bearer eyJhbGciOi..." \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Updated Sales Dashboard",
    "size": "extra_large"
  }'
```

#### Response

**Status**: `200 OK`

```json
{
  "id": "widget_123abc",
  "name": "Updated Sales Dashboard",
  "description": "Real-time sales metrics",
  "size": "extra_large",
  "updated_at": "2024-01-31T13:00:00Z"
}
```

---

### Delete Widget

Delete a widget.

```
DELETE /widgets/{widget_id}
```

#### Request

**Headers**:

```
Authorization: Bearer YOUR_ACCESS_TOKEN
```

**Required Scope**: `widgets:delete`

**Path Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `widget_id` | string | Yes | Widget identifier |

#### Example Request

```bash
curl -X DELETE "https://api.lazi.app/public/widgets/widget_123abc" \
  -H "Authorization: Bearer eyJhbGciOi..."
```

**Python Example**:

```python
import httpx

async def delete_widget(token: str, widget_id: str):
    url = f"https://api.lazi.app/public/widgets/{widget_id}"
    
    headers = {"Authorization": f"Bearer {token}"}
    
    async with httpx.AsyncClient() as client:
        response = await client.delete(url, headers=headers)
        return response.status_code == 204
```

#### Response

**Status**: `204 No Content`

---

### Attach Media

Upload and attach media files to a widget.

```
POST /widgets/{widget_id}/media
```

#### Request

**Headers**:

```
Authorization: Bearer YOUR_ACCESS_TOKEN
Content-Type: multipart/form-data
```

**Required Scope**: `widgets:write`, `files:write`

**Path Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `widget_id` | string | Yes | Widget identifier |

**Form Data**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Media file to upload |
| `object_name` | string | No | Custom object name (default: original filename) |

**Supported File Types**:
- Images: jpg, jpeg, png, gif, webp
- Videos: mp4, webm, mov
- Documents: pdf
- Max file size: 10MB

#### Example Request

```bash
curl -X POST "https://api.lazi.app/public/widgets/widget_123abc/media" \
  -H "Authorization: Bearer eyJhbGciOi..." \
  -F "file=@/path/to/image.png" \
  -F "object_name=dashboard_logo.png"
```

**Python Example**:

```python
from lazi_sdk import WriteWidget

async def attach_logo():
    widget = WriteWidget(
        token="eyJhbGciOi...",
        user_id="user_789xyz"
    )
    
    # Load existing widget
    await widget.create_new(
        name="Existing Widget",
        project_id="proj_123abc"
    )
    widget.current_widget.id = "widget_123abc"
    
    # Attach media
    await widget.attach_media(
        object_name="dashboard_logo.png",
        filename="./assets/logo.png"
    )
    
    await widget.save()
```

#### Response

**Status**: `200 OK`

```json
{
  "message": "Media attached successfully",
  "url": "https://s3.amazonaws.com/bucket/presigned-url",
  "object_name": "dashboard_logo.png",
  "expires_in": 3600
}
```

---

### Record Interaction

Record a user interaction with a widget (triggers custom logic execution).

```
POST /widgets/{widget_id}/interact
```

#### Request

**Headers**:

```
Authorization: Bearer YOUR_ACCESS_TOKEN
Content-Type: application/json
```

**Path Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `widget_id` | string | Yes | Widget identifier |

**Body**:

```json
{
  "endpoint": "https://api.example.com/sales",
  "params": {
    "period": "monthly",
    "metric": "revenue"
  }
}
```

#### Example Request

```bash
curl -X POST "https://api.lazi.app/public/widgets/widget_123abc/interact" \
  -H "Authorization: Bearer eyJhbGciOi..." \
  -H "Content-Type: application/json" \
  -d '{
    "endpoint": "https://api.example.com/sales",
    "params": {"period": "monthly"}
  }'
```

#### Response

**Status**: `200 OK`

```json
{
  "processed_data": {
    "revenue": 125000.50,
    "growth": 15.3,
    "status": "excellent"
  },
  "endpoint": "https://api.example.com/sales",
  "executed_at": "2024-01-31T12:00:00Z"
}
```

---

## Widget Size Options

| Size | Dimensions | Use Case |
|------|-----------|----------|
| `small` | 1x1 grid | Quick stats, icons |
| `medium` | 2x2 grid | Charts, small tables |
| `large` | 4x2 grid | Dashboards, detailed data |
| `extra_large` | 4x4 grid | Full dashboards, galleries |

---

## Custom Logic

### Logic Function Signature

```python
def process_data(widget_data: dict, request: Request, user: User) -> dict:
    """
    Process widget data from external endpoint.
    
    Args:
        widget_data: Data returned from the endpoint
        request: FastAPI Request object
        user: Current user object
    
    Returns:
        Processed data to display in widget
    """
    # Your custom logic here
    return processed_data
```

### Available Modules

Your custom logic can import:
- `datetime`, `time`
- `json`, `re`
- `math`, `statistics`
- `collections`, `itertools`

**Forbidden modules** (for security):
- `os`, `sys`, `subprocess`
- `eval`, `exec`, `compile`
- `open`, `file`
- `import`, `__import__`

### Example Logic Functions

**Calculate Percentage Change**:

```python
def process_data(widget_data, request, user):
    current = widget_data.get('current_value', 0)
    previous = widget_data.get('previous_value', 1)
    
    change = ((current - previous) / previous) * 100
    
    return {
        'current': current,
        'change_percent': round(change, 2),
        'trend': 'up' if change > 0 else 'down'
    }
```

**Filter and Sort Data**:

```python
def process_data(widget_data, request, user):
    items = widget_data.get('items', [])
    
    # Filter active items
    active_items = [item for item in items if item.get('status') == 'active']
    
    # Sort by priority
    sorted_items = sorted(active_items, key=lambda x: x.get('priority', 0), reverse=True)
    
    return {
        'total': len(items),
        'active': len(active_items),
        'top_items': sorted_items[:5]
    }
```

---

## Error Responses

### 400 Bad Request

```json
{
  "detail": "Invalid widget size. Must be one of: small, medium, large, extra_large",
  "error_code": "INVALID_SIZE"
}
```

### 403 Forbidden

```json
{
  "detail": "Insufficient permissions. Required scope: widgets:write",
  "error_code": "INSUFFICIENT_SCOPE"
}
```

### 404 Not Found

```json
{
  "detail": "Widget not found",
  "error_code": "WIDGET_NOT_FOUND"
}
```

### 422 Validation Error

```json
{
  "detail": [
    {
      "loc": ["body", "name"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

---

## Rate Limits

- **Widget creation**: 100 per hour per user
- **Widget updates**: 500 per hour per user
- **Media uploads**: 50 per hour per user
- **Interactions**: 1000 per hour per user

---

## Next Steps

- [Projects API](projects.md) - Manage projects
- [Custom Endpoints](../advanced/custom-endpoints.md) - Create dynamic widget endpoints
- [Examples](../sdk/examples.md) - Complete widget examples

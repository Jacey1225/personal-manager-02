# Projects API Reference

Project management endpoints for the Lazi Public API.

## Overview

Projects are containers for organizing widgets, pages, and team members. Each project can have multiple widgets and can be public or private.

**Base Path**: `/public/projects`

**Required Scopes**: 
- `projects:read` - Read project data
- `projects:write` - Create/update projects
- `projects:delete` - Delete projects

## Endpoints

### List Projects

Retrieve a list of projects accessible to the authenticated user.

```
GET /projects
```

#### Request

**Headers**:

```
Authorization: Bearer YOUR_ACCESS_TOKEN
```

**Query Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `page` | integer | No | Page number (default: 1) |
| `per_page` | integer | No | Items per page (default: 20, max: 100) |
| `transparency` | boolean | No | Filter by visibility: true (public), false (private) |
| `organization_id` | string | No | Filter by organization |
| `sort` | string | No | Sort field and order (e.g., "created_at:desc") |

#### Example Request

```bash
curl -X GET "https://api.lazi.app/public/projects?per_page=10&transparency=true" \
  -H "Authorization: Bearer eyJhbGciOi..."
```

**Python Example**:

```python
import httpx

async def list_projects(token: str):
    url = "https://api.lazi.app/public/projects"
    
    headers = {"Authorization": f"Bearer {token}"}
    params = {"per_page": 20, "transparency": True}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, params=params)
        return response.json()
```

#### Response

**Status**: `200 OK`

```json
{
  "data": [
    {
      "project_id": "proj_123abc",
      "project_name": "Sales Analytics Platform",
      "project_likes": 42,
      "project_transparency": true,
      "project_members": [
        "user_789xyz",
        "user_456def"
      ],
      "organizations": [
        "org_111aaa"
      ],
      "created_at": "2024-01-15T10:00:00Z",
      "updated_at": "2024-01-31T12:00:00Z",
      "widget_count": 8
    },
    {
      "project_id": "proj_456def",
      "project_name": "Internal Dashboard",
      "project_likes": 15,
      "project_transparency": false,
      "project_members": [
        "user_789xyz"
      ],
      "organizations": [],
      "created_at": "2024-01-20T14:30:00Z",
      "updated_at": "2024-01-30T09:15:00Z",
      "widget_count": 3
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 10,
    "total_items": 12,
    "total_pages": 2,
    "has_next": true,
    "has_previous": false
  }
}
```

---

### Get Project

Retrieve details of a specific project.

```
GET /projects/{project_id}
```

#### Request

**Headers**:

```
Authorization: Bearer YOUR_ACCESS_TOKEN
```

**Path Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `project_id` | string | Yes | Project identifier |

#### Example Request

```bash
curl -X GET "https://api.lazi.app/public/projects/proj_123abc" \
  -H "Authorization: Bearer eyJhbGciOi..."
```

**Python Example**:

```python
import httpx

async def get_project(token: str, project_id: str):
    url = f"https://api.lazi.app/public/projects/{project_id}"
    
    headers = {"Authorization": f"Bearer {token}"}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        return response.json()
```

#### Response

**Status**: `200 OK`

```json
{
  "project_id": "proj_123abc",
  "project_name": "Sales Analytics Platform",
  "project_likes": 42,
  "project_transparency": true,
  "project_members": [
    "user_789xyz",
    "user_456def",
    "user_222bbb"
  ],
  "organizations": [
    "org_111aaa"
  ],
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-31T12:00:00Z",
  "metadata": {
    "widget_count": 8,
    "total_interactions": 1250,
    "last_activity": "2024-01-31T11:45:00Z"
  }
}
```

---

### Create Project

Create a new project.

```
POST /projects
```

#### Request

**Headers**:

```
Authorization: Bearer YOUR_ACCESS_TOKEN
Content-Type: application/json
```

**Required Scope**: `projects:write`

**Body**:

```json
{
  "project_name": "Marketing Dashboard",
  "project_transparency": true,
  "project_members": ["user_789xyz"],
  "organizations": ["org_111aaa"]
}
```

**Request Schema**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `project_name` | string | Yes | Project name |
| `project_transparency` | boolean | No | Public (true) or private (false). Default: true |
| `project_members` | array | Yes | Array of user IDs (must include creator) |
| `organizations` | array | No | Array of organization IDs |

#### Example Request

```bash
curl -X POST "https://api.lazi.app/public/projects" \
  -H "Authorization: Bearer eyJhbGciOi..." \
  -H "Content-Type: application/json" \
  -d '{
    "project_name": "Marketing Dashboard",
    "project_transparency": true,
    "project_members": ["user_789xyz"],
    "organizations": []
  }'
```

**Python Example**:

```python
import httpx

async def create_project(token: str, user_id: str):
    url = "https://api.lazi.app/public/projects"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    data = {
        "project_name": "Marketing Dashboard",
        "project_transparency": True,
        "project_members": [user_id],
        "organizations": []
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=data)
        return response.json()
```

#### Response

**Status**: `201 Created`

```json
{
  "project_id": "proj_789ghi",
  "project_name": "Marketing Dashboard",
  "project_likes": 0,
  "project_transparency": true,
  "project_members": ["user_789xyz"],
  "organizations": [],
  "created_at": "2024-01-31T12:00:00Z",
  "updated_at": "2024-01-31T12:00:00Z"
}
```

---

### Update Project

Update an existing project.

```
PATCH /projects/{project_id}
```

#### Request

**Headers**:

```
Authorization: Bearer YOUR_ACCESS_TOKEN
Content-Type: application/json
```

**Required Scope**: `projects:write`

**Path Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `project_id` | string | Yes | Project identifier |

**Body**: (All fields optional, include only fields to update)

```json
{
  "project_name": "Updated Project Name",
  "project_transparency": false,
  "project_members": ["user_789xyz", "user_111new"]
}
```

#### Example Request

```bash
curl -X PATCH "https://api.lazi.app/public/projects/proj_123abc" \
  -H "Authorization: Bearer eyJhbGciOi..." \
  -H "Content-Type: application/json" \
  -d '{
    "project_name": "Sales & Marketing Analytics",
    "project_transparency": false
  }'
```

#### Response

**Status**: `200 OK`

```json
{
  "project_id": "proj_123abc",
  "project_name": "Sales & Marketing Analytics",
  "project_transparency": false,
  "updated_at": "2024-01-31T13:00:00Z"
}
```

---

### Delete Project

Delete a project and all associated widgets.

```
DELETE /projects/{project_id}
```

#### Request

**Headers**:

```
Authorization: Bearer YOUR_ACCESS_TOKEN
```

**Required Scope**: `projects:delete`

**Path Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `project_id` | string | Yes | Project identifier |

**Query Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `confirm` | boolean | Yes | Must be true to confirm deletion |

#### Example Request

```bash
curl -X DELETE "https://api.lazi.app/public/projects/proj_123abc?confirm=true" \
  -H "Authorization: Bearer eyJhbGciOi..."
```

**Python Example**:

```python
import httpx

async def delete_project(token: str, project_id: str):
    url = f"https://api.lazi.app/public/projects/{project_id}"
    
    headers = {"Authorization": f"Bearer {token}"}
    params = {"confirm": True}
    
    async with httpx.AsyncClient() as client:
        response = await client.delete(url, headers=headers, params=params)
        return response.status_code == 204
```

#### Response

**Status**: `204 No Content`

---

### List Project Widgets

Retrieve all widgets for a specific project.

```
GET /projects/{project_id}/widgets
```

#### Request

**Headers**:

```
Authorization: Bearer YOUR_ACCESS_TOKEN
```

**Path Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `project_id` | string | Yes | Project identifier |

**Query Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `page` | integer | No | Page number (default: 1) |
| `per_page` | integer | No | Items per page (default: 20, max: 100) |
| `size` | string | No | Filter by widget size |

#### Example Request

```bash
curl -X GET "https://api.lazi.app/public/projects/proj_123abc/widgets?per_page=50" \
  -H "Authorization: Bearer eyJhbGciOi..."
```

**Python Example**:

```python
from lazi_sdk import ReadWidget

async def get_project_widgets():
    reader = ReadWidget(
        username="john@example.com",
        token="eyJhbGciOi...",
        project_id="proj_123abc"
    )
    
    widgets = await reader.list_widgets()
    
    print(f"Total widgets: {len(widgets)}")
    for widget in widgets:
        print(f"- {widget.name} ({widget.size})")
```

#### Response

**Status**: `200 OK`

```json
{
  "data": [
    {
      "id": "widget_123abc",
      "name": "Sales Dashboard",
      "size": "large",
      "created_at": "2024-01-20T10:00:00Z"
    },
    {
      "id": "widget_456def",
      "name": "User Analytics",
      "size": "medium",
      "created_at": "2024-01-22T14:30:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 50,
    "total_items": 8,
    "total_pages": 1
  }
}
```

---

### Add Project Member

Add a user to a project.

```
POST /projects/{project_id}/members
```

#### Request

**Headers**:

```
Authorization: Bearer YOUR_ACCESS_TOKEN
Content-Type: application/json
```

**Required Scope**: `projects:write`

**Path Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `project_id` | string | Yes | Project identifier |

**Body**:

```json
{
  "user_id": "user_999zzz",
  "role": "contributor"
}
```

**Roles**:
- `owner` - Full project control
- `admin` - Manage members and settings
- `contributor` - Create and edit widgets
- `viewer` - Read-only access

#### Example Request

```bash
curl -X POST "https://api.lazi.app/public/projects/proj_123abc/members" \
  -H "Authorization: Bearer eyJhbGciOi..." \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_999zzz",
    "role": "contributor"
  }'
```

#### Response

**Status**: `200 OK`

```json
{
  "message": "Member added successfully",
  "project_id": "proj_123abc",
  "user_id": "user_999zzz",
  "role": "contributor"
}
```

---

### Remove Project Member

Remove a user from a project.

```
DELETE /projects/{project_id}/members/{user_id}
```

#### Request

**Headers**:

```
Authorization: Bearer YOUR_ACCESS_TOKEN
```

**Required Scope**: `projects:write`

**Path Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `project_id` | string | Yes | Project identifier |
| `user_id` | string | Yes | User identifier to remove |

#### Example Request

```bash
curl -X DELETE "https://api.lazi.app/public/projects/proj_123abc/members/user_999zzz" \
  -H "Authorization: Bearer eyJhbGciOi..."
```

#### Response

**Status**: `204 No Content`

---

### Like Project

Add a like to a project (increment like count).

```
POST /projects/{project_id}/like
```

#### Request

**Headers**:

```
Authorization: Bearer YOUR_ACCESS_TOKEN
```

**Required Scope**: `projects:read`

**Path Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `project_id` | string | Yes | Project identifier |

#### Example Request

```bash
curl -X POST "https://api.lazi.app/public/projects/proj_123abc/like" \
  -H "Authorization: Bearer eyJhbGciOi..."
```

#### Response

**Status**: `200 OK`

```json
{
  "project_id": "proj_123abc",
  "project_likes": 43,
  "liked_by_user": true
}
```

---

### Unlike Project

Remove a like from a project (decrement like count).

```
DELETE /projects/{project_id}/like
```

#### Request

**Headers**:

```
Authorization: Bearer YOUR_ACCESS_TOKEN
```

**Required Scope**: `projects:read`

**Path Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `project_id` | string | Yes | Project identifier |

#### Example Request

```bash
curl -X DELETE "https://api.lazi.app/public/projects/proj_123abc/like" \
  -H "Authorization: Bearer eyJhbGciOi..."
```

#### Response

**Status**: `200 OK`

```json
{
  "project_id": "proj_123abc",
  "project_likes": 42,
  "liked_by_user": false
}
```

---

## Project Transparency

### Public Projects (`transparency: true`)

- Visible to all users
- Can be discovered via search
- Anyone with link can view widgets
- Only members can edit

### Private Projects (`transparency: false`)

- Visible only to members
- Not discoverable via search
- Requires authentication and membership
- Members only can view and edit

---

## Error Responses

### 400 Bad Request

```json
{
  "detail": "Project name is required",
  "error_code": "MISSING_FIELD"
}
```

### 403 Forbidden

```json
{
  "detail": "You are not a member of this project",
  "error_code": "NOT_PROJECT_MEMBER"
}
```

### 404 Not Found

```json
{
  "detail": "Project not found",
  "error_code": "PROJECT_NOT_FOUND"
}
```

### 409 Conflict

```json
{
  "detail": "User is already a member of this project",
  "error_code": "MEMBER_EXISTS"
}
```

---

## Rate Limits

- **Project creation**: 20 per hour per user
- **Project updates**: 100 per hour per user
- **Member operations**: 50 per hour per project
- **Like operations**: 200 per hour per user

---

## Complete Example

Create a project with widgets:

```python
import asyncio
import os
from lazi_sdk import WriteWidget
import httpx

async def create_complete_project():
    token = os.getenv("LAZI_TOKEN")
    user_id = os.getenv("LAZI_USER_ID")
    
    # 1. Create project
    url = "https://api.lazi.app/public/projects"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    project_data = {
        "project_name": "Analytics Hub",
        "project_transparency": True,
        "project_members": [user_id],
        "organizations": []
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=project_data)
        project = response.json()
        project_id = project["project_id"]
    
    print(f"✅ Project created: {project_id}")
    
    # 2. Create widgets
    widget_configs = [
        {"name": "Revenue", "size": "medium"},
        {"name": "Users", "size": "medium"},
        {"name": "Activity", "size": "large"}
    ]
    
    for config in widget_configs:
        widget = WriteWidget(token=token, user_id=user_id)
        
        await widget.create_new(
            name=config["name"],
            size=config["size"],
            project_id=project_id
        )
        
        await widget.save()
        print(f"✅ Widget created: {config['name']}")
    
    print(f"\n🎉 Complete! Project with {len(widget_configs)} widgets")

asyncio.run(create_complete_project())
```

---

## Next Steps

- [Widgets API](widgets.md) - Manage widgets
- [Authentication](authentication.md) - OAuth2 authentication
- [SDK Examples](../sdk/examples.md) - Complete examples

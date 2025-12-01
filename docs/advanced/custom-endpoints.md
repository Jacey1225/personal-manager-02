# Custom Endpoints

Guide to creating dynamic, user-defined API endpoints in Lazi.

## Overview

The Lazi API allows users to create custom endpoints for their widgets that execute user-defined Python logic. This enables powerful, flexible widget behaviors while maintaining security through code sandboxing.

## How It Works

### 1. User Creates Widget with Endpoint

```python
from lazi_sdk import WriteWidget

widget = WriteWidget(token="...", user_id="...")

await widget.create_new(
    name="Sales Dashboard",
    project_id="proj_123"
)

await widget.post(
    endpoint="https://api.example.com/sales",
    endpoint_data={
        "params": {"period": "weekly"},
        "headers": {"Authorization": "Bearer token"},
        "refresh_interval": 300,
        "logic": """
def process_data(widget_data, request, user):
    # Custom processing logic
    revenue = widget_data.get('revenue', 0)
    return {'revenue_formatted': f'${revenue:,.2f}'}
        """
    }
)

await widget.save()
```

### 2. API Registers Dynamic Route

When the widget is created, the API automatically registers a new endpoint:

```
GET /public/widgets/{widget_id}/execute
```

### 3. Endpoint Execution Flow

```
User Request → Widget Endpoint → External API → Custom Logic → Response
     ↓              ↓                   ↓              ↓            ↓
   Browser      FastAPI Route      Fetch Data     Sandbox     Formatted Data
```

## Custom Logic

### Function Signature

All custom logic must define a `process_data` function:

```python
def process_data(widget_data: dict, request: Request, user: User) -> dict:
    """
    Process data from external endpoint.
    
    Args:
        widget_data: JSON response from the external endpoint
        request: FastAPI Request object (read-only)
        user: User object with id, username, email
    
    Returns:
        dict: Processed data to return to client
    """
    # Your logic here
    return processed_data
```

### Available Parameters

#### 1. `widget_data`

The JSON response from the external endpoint:

```python
def process_data(widget_data, request, user):
    # Access external API data
    sales = widget_data.get('sales', [])
    total = sum(item['amount'] for item in sales)
    
    return {'total_sales': total}
```

#### 2. `request`

FastAPI Request object (read-only access):

```python
def process_data(widget_data, request, user):
    # Access request headers
    user_agent = request.headers.get('user-agent')
    
    # Access query parameters
    filter_param = request.query_params.get('filter')
    
    return {'filtered': filter_param in widget_data}
```

#### 3. `user`

Current user object:

```python
def process_data(widget_data, request, user):
    # Access user information
    user_id = user.id
    username = user.username
    email = user.email
    
    # Customize response per user
    return {
        'data': widget_data,
        'personalized_message': f'Hello, {username}!'
    }
```

## Security Sandbox

### Allowed Modules

Your custom logic can import these standard library modules:

```python
# ✅ Allowed
import datetime
import time
import json
import re
import math
import statistics
import collections
import itertools
from typing import Any, List, Dict
```

### Forbidden Modules

These modules are **blocked** for security:

```python
# ❌ Forbidden - File system access
import os
import sys
import pathlib
import shutil
import glob

# ❌ Forbidden - Network access
import socket
import requests
import urllib
import http

# ❌ Forbidden - Process control
import subprocess
import multiprocessing
import threading
import signal

# ❌ Forbidden - Arbitrary code execution
eval()
exec()
compile()
__import__()

# ❌ Forbidden - Cloud SDKs
import boto3
import google.cloud
import azure
```

### Resource Limits

Custom logic execution is constrained:

| Limit | Value | Reason |
|-------|-------|--------|
| **Execution Time** | 5 seconds | Prevent infinite loops |
| **Memory** | 1024 MB | Prevent memory exhaustion |
| **CPU** | Single core | Isolate from other requests |
| **Recursion Depth** | 100 | Prevent stack overflow |
| **Nesting Depth** | 10 | Prevent deeply nested loops |

### Code Validation

Before execution, code is validated using AST parsing:

```python
import ast

def validate_code(code: str) -> List[str]:
    """
    Validate Python code for forbidden imports/calls.
    
    Returns:
        List of forbidden modules/functions found
    """
    tree = ast.parse(code)
    validator = ValidateModules()
    validator.visit(tree)
    return validator.found  # [] if valid, [forbidden_items] if invalid
```

### Execution Sandbox

Code runs in an isolated process with resource limits:

```python
import multiprocessing as mp
import resource
import signal

def run_sandboxed(code: str, context: dict, timeout: float = 5.0):
    """
    Execute code in isolated process with timeout and memory limits.
    """
    def target(queue: mp.Queue):
        # Set memory limit (1024 MB)
        resource.setrlimit(
            resource.RLIMIT_AS,
            (1024 * 1024 * 1024, 1024 * 1024 * 1024)
        )
        
        try:
            # Execute code with context
            exec(code, context)
            result = context.get('process_data')(
                context['widget_data'],
                context['request'],
                context['user']
            )
            queue.put({'success': True, 'result': result})
        except Exception as e:
            queue.put({'success': False, 'error': str(e)})
    
    queue = mp.Queue()
    process = mp.Process(target=target, args=(queue,))
    process.start()
    process.join(timeout=timeout)
    
    if process.is_alive():
        process.terminate()
        raise TimeoutError("Code execution exceeded time limit")
    
    return queue.get()
```

## Dynamic Route Registration

### How Routes Are Created

When a widget with custom logic is saved:

```python
from fastapi import APIRouter, FastAPI

app = FastAPI()

# User creates widget with custom endpoint
widget_id = "widget_123abc"
endpoint_url = "https://api.example.com/sales"
custom_logic = "def process_data(data, req, user): return data"

# API dynamically registers route
@app.get(f"/public/widgets/{widget_id}/execute")
async def execute_widget(request: Request):
    # 1. Fetch data from endpoint_url
    async with httpx.AsyncClient() as client:
        response = await client.get(endpoint_url)
        widget_data = response.json()
    
    # 2. Execute custom logic in sandbox
    result = run_sandboxed(custom_logic, {
        'widget_data': widget_data,
        'request': request,
        'user': current_user
    })
    
    # 3. Return processed data
    return result
```

### Route Lifecycle

```
Widget Created → Validate Logic → Register Route → Route Active
                      ↓                ↓               ↓
                  AST Check     app.add_api_route()  Accept Requests

Widget Deleted → Unregister Route → Route Removed
                      ↓                    ↓
                Remove from app     404 for requests
```

### Limitations

- Routes are **not** reflected in OpenAPI schema automatically
- Routes persist only during application runtime (no database persistence)
- Restarting the server requires re-loading routes from database

## Advanced Examples

### Example 1: Data Transformation

```python
def process_data(widget_data, request, user):
    """Transform sales data into chart format"""
    sales = widget_data.get('sales', [])
    
    # Group by month
    from collections import defaultdict
    monthly = defaultdict(float)
    
    for sale in sales:
        month = sale['date'][:7]  # YYYY-MM
        monthly[month] += sale['amount']
    
    # Format for chart
    return {
        'labels': list(monthly.keys()),
        'data': list(monthly.values()),
        'total': sum(monthly.values())
    }
```

### Example 2: Filtering and Sorting

```python
def process_data(widget_data, request, user):
    """Filter and sort items based on user preference"""
    items = widget_data.get('items', [])
    
    # Get user preference from request
    sort_by = request.query_params.get('sort', 'priority')
    filter_status = request.query_params.get('status', 'all')
    
    # Filter
    if filter_status != 'all':
        items = [i for i in items if i.get('status') == filter_status]
    
    # Sort
    items.sort(key=lambda x: x.get(sort_by, 0), reverse=True)
    
    return {
        'items': items[:10],  # Top 10
        'total': len(items),
        'sorted_by': sort_by
    }
```

### Example 3: Statistical Analysis

```python
def process_data(widget_data, request, user):
    """Calculate statistics on numeric data"""
    import statistics
    
    values = [item['value'] for item in widget_data.get('data', [])]
    
    if not values:
        return {'error': 'No data available'}
    
    return {
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'stdev': statistics.stdev(values) if len(values) > 1 else 0,
        'min': min(values),
        'max': max(values),
        'count': len(values)
    }
```

### Example 4: Time-based Processing

```python
def process_data(widget_data, request, user):
    """Process time-series data"""
    from datetime import datetime, timedelta
    
    events = widget_data.get('events', [])
    
    # Get events from last 24 hours
    now = datetime.utcnow()
    cutoff = now - timedelta(hours=24)
    
    recent_events = [
        e for e in events
        if datetime.fromisoformat(e['timestamp']) > cutoff
    ]
    
    return {
        'total_events': len(events),
        'recent_events': len(recent_events),
        'recent_percentage': (len(recent_events) / len(events) * 100) if events else 0
    }
```

### Example 5: User-Personalized Data

```python
def process_data(widget_data, request, user):
    """Personalize widget data per user"""
    all_items = widget_data.get('items', [])
    
    # Filter items relevant to current user
    user_items = [
        item for item in all_items
        if user.id in item.get('assigned_users', [])
    ]
    
    # Calculate user-specific metrics
    total_value = sum(item.get('value', 0) for item in user_items)
    
    return {
        'user_id': user.id,
        'username': user.username,
        'assigned_items': len(user_items),
        'total_value': total_value,
        'items': user_items[:5]  # Top 5
    }
```

## Error Handling

### Validation Errors

```python
# Code with forbidden import
code = """
import os  # ❌ Forbidden!

def process_data(data, req, user):
    return os.listdir('/')  # Security risk
"""

# Validation catches this
forbidden = validate_code(code)
# Returns: ['os']

# Error response
{
    "detail": "Code validation failed: Forbidden module 'os' detected",
    "error_code": "FORBIDDEN_MODULE",
    "forbidden_modules": ["os"]
}
```

### Runtime Errors

```python
# Code with runtime error
code = """
def process_data(data, req, user):
    return data['missing_key']  # KeyError!
"""

# Sandbox catches exception
# Error response
{
    "detail": "Code execution error: 'missing_key'",
    "error_code": "EXECUTION_ERROR",
    "error_type": "KeyError"
}
```

### Timeout Errors

```python
# Code with infinite loop
code = """
def process_data(data, req, user):
    while True:  # ❌ Infinite loop
        pass
    return {}
"""

# Sandbox terminates after 5 seconds
# Error response
{
    "detail": "Code execution exceeded time limit (5 seconds)",
    "error_code": "TIMEOUT_ERROR"
}
```

## Best Practices

### 1. Keep Logic Simple

```python
# ✅ Good - simple, focused
def process_data(data, req, user):
    return {'total': sum(data.get('values', []))}

# ❌ Bad - complex, error-prone
def process_data(data, req, user):
    # 100 lines of complex logic...
```

### 2. Handle Missing Data

```python
# ✅ Good - defensive programming
def process_data(data, req, user):
    items = data.get('items', [])
    if not items:
        return {'error': 'No data available'}
    
    return {'count': len(items)}

# ❌ Bad - assumes data exists
def process_data(data, req, user):
    return {'count': len(data['items'])}  # KeyError if missing
```

### 3. Validate Input

```python
def process_data(data, req, user):
    # Validate expected structure
    if not isinstance(data, dict):
        return {'error': 'Invalid data format'}
    
    if 'sales' not in data:
        return {'error': 'Missing sales data'}
    
    # Process validated data
    return {'total': sum(s['amount'] for s in data['sales'])}
```

### 4. Return Consistent Structure

```python
# ✅ Good - consistent response
def process_data(data, req, user):
    if error_condition:
        return {'success': False, 'error': 'Error message'}
    
    return {
        'success': True,
        'data': processed_data
    }

# ❌ Bad - inconsistent responses
def process_data(data, req, user):
    if error_condition:
        return None  # or raises exception
    
    return processed_data  # different structure
```

## Testing Custom Logic

### Local Testing

```python
# test_custom_logic.py
def test_process_data():
    # Mock data
    widget_data = {
        'sales': [
            {'amount': 100, 'date': '2024-01'},
            {'amount': 200, 'date': '2024-02'}
        ]
    }
    
    # Mock request and user
    class MockRequest:
        query_params = {}
    
    class MockUser:
        id = 'user_123'
        username = 'test_user'
    
    # Execute logic
    code = """
def process_data(data, req, user):
    total = sum(s['amount'] for s in data.get('sales', []))
    return {'total': total}
    """
    
    context = {
        'widget_data': widget_data,
        'request': MockRequest(),
        'user': MockUser()
    }
    
    exec(code, context)
    result = context['process_data'](widget_data, MockRequest(), MockUser())
    
    assert result == {'total': 300}
```

### Integration Testing

```python
import httpx
import pytest

@pytest.mark.asyncio
async def test_widget_endpoint():
    # Create widget with custom logic
    widget_id = await create_test_widget()
    
    # Execute endpoint
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://api.lazi.app/public/widgets/{widget_id}/execute",
            headers={"Authorization": f"Bearer {test_token}"}
        )
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert 'total' in data
```

## Next Steps

- [Security Best Practices](security.md) - Security guidelines
- [OAuth2 Scopes](scopes.md) - Permission management
- [SDK Examples](../sdk/examples.md) - Complete examples

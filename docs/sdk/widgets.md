# Widget Management with SDK

Complete guide to creating, reading, updating, and managing widgets using the Lazi SDK.

## Widget Classes

The SDK provides two main classes for widget management:

- **`WriteWidget`** - Create, modify, and delete widgets
- **`ReadWidget`** - Read and list widgets

## Creating Widgets

### Basic Widget Creation

```python
from lazi_sdk import WriteWidget

# Initialize
widget = WriteWidget(
    token="your_access_token",
    user_id="user_123"
)

# Create a new widget
await widget.create_new(
    name="Sales Dashboard",
    description="Real-time sales metrics",
    size="large",  # Options: small, medium, large, extra_large
    project_id="project_456"
)

# Save to database
await widget.save()

print(f"Widget created with ID: {widget.current_widget.id}")
```

### Widget Sizes

Available widget sizes:

- `"small"` - Compact widget (1x1 grid)
- `"medium"` - Standard widget (2x2 grid)
- `"large"` - Large widget (4x2 grid)
- `"extra_large"` - Extra large widget (4x4 grid)

## Attaching Media

### Upload Files to Widgets

```python
# Create widget
await widget.create_new(
    name="Product Gallery",
    description="Product images",
    project_id="project_001"
)

# Attach image
await widget.attach_media(
    object_name="product_image.jpg",
    filename="./local/path/to/image.jpg"
)

# Attach multiple files
for image in ["img1.jpg", "img2.jpg", "img3.jpg"]:
    await widget.attach_media(
        object_name=image,
        filename=f"./images/{image}"
    )

# Save with all media
await widget.save()
```

### Supported Media Types

- **Images**: JPG, PNG, GIF, WebP
- **Videos**: MP4, WebM, AVI
- **Documents**: PDF, DOCX, TXT
- **Archives**: ZIP, TAR

## Widget Interactions

### Configure Endpoint and Logic

```python
# Create widget with custom interaction
await widget.create_new(
    name="GitHub Stats",
    description="Repository statistics",
    project_id="project_001"
)

# Configure interaction endpoint
await widget.post(
    endpoint="https://api.github.com/repos/owner/repo",
    endpoint_data={
        "params": {
            "type": "all",
            "sort": "updated"
        },
        "headers": {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": "token ghp_xxxxx"
        },
        "refresh_interval": 1800,  # Refresh every 30 minutes
        "logic": """
def process_data(widget_data, request, user):
    stars = widget_data.get('stargazers_count', 0)
    forks = widget_data.get('forks_count', 0)
    return {
        'stars': stars,
        'forks': forks,
        'popularity': 'high' if stars > 1000 else 'moderate'
    }
        """
    }
)

await widget.save()
```

### Interaction Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `endpoint` | `str` | API endpoint URL |
| `params` | `Dict[str, Any]` | Query parameters |
| `headers` | `Dict[str, str]` | HTTP headers |
| `refresh_interval` | `int` | Auto-refresh interval in seconds (0 = no refresh) |
| `logic` | `str` | Python code to process response data |

## Reading Widgets

### List All Widgets in Project

```python
from lazi_sdk import ReadWidget

# Initialize reader
reader = ReadWidget(
    username="developer@example.com",
    token="read_access_token",
    project_id="project_123"
)

# List all widgets
widgets = await reader.list_widgets()

for widget in widgets:
    print(f"Widget: {widget['name']} (ID: {widget['widget_id']})")
```

### Get Specific Widget

```python
# Get widget by ID
widget_data = await reader.get_widget("widget_abc123")

print(f"Widget Name: {widget_data['name']}")
print(f"Description: {widget_data['description']}")
print(f"Size: {widget_data['size']}")
print(f"Content URLs: {widget_data['content']}")
```

## Advanced Widget Creation

### Using Custom Decorator

```python
from lazi_sdk import WriteWidget

widget = WriteWidget(token=token, user_id=user_id)

# Use decorator pattern for custom creation logic
@widget.create
def build_custom_widget():
    return {
        "name": "Custom Analytics",
        "description": "Advanced metrics dashboard",
        "size": "large",
        "content": [],
        "interaction": {
            "params": {"metric": "revenue"},
            "headers": {"Authorization": "Bearer token"},
            "refresh_interval": 300
        }
    }

# Execute decorator
result = build_custom_widget()

# Save
await widget.save()
```

## Widget Updates

### Modify Existing Widget

```python
# Get widget
reader = ReadWidget(username=username, token=token, project_id=project_id)
widget_data = await reader.get_widget("widget_123")

# Initialize writer with existing widget
writer = WriteWidget(token=token, user_id=user_id)
writer.current_widget = WidgetConfig(**widget_data)
writer.project_id = project_id

# Update properties
writer.current_widget.name = "Updated Dashboard"
writer.current_widget.description = "New description"

# Save changes
await writer.save()
```

## Complete Example

### Building a Weather Widget

```python
from lazi_sdk import WriteWidget
import os

async def create_weather_widget():
    # Initialize
    widget = WriteWidget(
        token=os.getenv("LAZI_TOKEN"),
        user_id=os.getenv("LAZI_USER_ID")
    )
    
    # Create widget
    await widget.create_new(
        name="Weather Dashboard",
        description="Current weather conditions",
        size="medium",
        project_id="weather_project_001"
    )
    
    # Configure weather API endpoint
    await widget.post(
        endpoint="https://api.openweathermap.org/data/2.5/weather",
        endpoint_data={
            "params": {
                "q": "San Francisco",
                "appid": os.getenv("WEATHER_API_KEY"),
                "units": "metric"
            },
            "headers": {
                "Content-Type": "application/json"
            },
            "refresh_interval": 600,  # Refresh every 10 minutes
            "logic": """
def process_data(widget_data, request, user):
    temp = widget_data['main']['temp']
    feels_like = widget_data['main']['feels_like']
    humidity = widget_data['main']['humidity']
    description = widget_data['weather'][0]['description']
    
    return {
        'temperature': f"{temp}°C",
        'feels_like': f"{feels_like}°C",
        'humidity': f"{humidity}%",
        'conditions': description.title(),
        'last_updated': widget_data['dt']
    }
            """
        }
    )
    
    # Attach weather icon
    await widget.attach_media(
        object_name="weather_icon.png",
        filename="./assets/weather_icon.png"
    )
    
    # Save
    await widget.save()
    
    print(f"✅ Weather widget created: {widget.current_widget.id}")
    return widget.current_widget.id

# Run
import asyncio
asyncio.run(create_weather_widget())
```

## Error Handling

```python
from fastapi import HTTPException

async def safe_widget_creation():
    try:
        widget = WriteWidget(token=token, user_id=user_id)
        
        await widget.create_new(
            name="Safe Widget",
            description="With error handling",
            project_id="project_001"
        )
        
        await widget.save()
        
    except HTTPException as e:
        if e.status_code == 403:
            print("Permission denied: Check your scopes")
        elif e.status_code == 404:
            print("Project not found")
        else:
            print(f"Error: {e.detail}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
```

## Next Steps

- [Examples](examples.md) - More complete widget examples
- [Custom Endpoints](../advanced/custom-endpoints.md) - Create custom API endpoints for widgets
- [API Reference](../api/widgets.md) - Detailed API documentation

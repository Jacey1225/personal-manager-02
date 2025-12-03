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
from api.schemas.widgets import WidgetSize

# Initialize SDK with OAuth token
widget = WriteWidget(
    username="developer@example.com",
    token="your_access_token",
    project_id="project_456"
)

# Create a new widget
await widget.create(
    name="Sales Dashboard",
    size=WidgetSize.LARGE.value  # Options: small, medium, large, extra_large
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

## Uploading Media Files

### Upload Files to S3 with Presigned URLs

```python
from api.schemas.widgets import WidgetSize

# Create widget
await widget.create(
    name="Product Gallery",
    size=WidgetSize.LARGE.value
)

# Define interaction
await widget.interaction(
    endpoint="/api/v1/widgets/gallery/view",
    headers={},
    refresh_interval=0,
    func=lambda data, req, user: data
)

# Upload image and get presigned URL
image_url = await widget.uploadable(
    object_name="product_image.jpg",
    filename="./local/path/to/image.jpg",
    expire=3600  # URL expires in 1 hour
)

print(f"Image URL: {image_url}")

# Upload multiple files
image_urls = []
for image in ["img1.jpg", "img2.jpg", "img3.jpg"]:
    url = await widget.uploadable(
        object_name=image,
        filename=f"./images/{image}",
        expire=3600
    )
    image_urls.append(url)

# Save widget
await widget.save()
```

### Supported Media Types

AWS S3 supports all file types:

- **Images**: JPG, PNG, GIF, WebP, SVG
- **Videos**: MP4, WebM, AVI, MOV
- **Documents**: PDF, DOCX, XLSX, TXT, CSV
- **Archives**: ZIP, TAR, GZ
- **Audio**: MP3, WAV, OGG
- **Any other file type**

### Presigned URL Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `object_name` | `str` | Name/key for the file in S3 |
| `filename` | `Optional[str]` | Local file path to upload (if not already in S3) |
| `expire` | `int` | URL expiration time in seconds (default: 3600) |

## Widget Interactions

### Configure Endpoint and Logic

```python
from api.schemas.widgets import WidgetSize

# Create widget
await widget.create(
    name="GitHub Stats",
    size=WidgetSize.MEDIUM.value
)

# Define the data processing function
def process_github_data(widget_data, request, user):
    stars = widget_data.get('stargazers_count', 0)
    forks = widget_data.get('forks_count', 0)
    return {
        'stars': stars,
        'forks': forks,
        'popularity': 'high' if stars > 1000 else 'moderate'
    }

# Configure interaction endpoint
await widget.interaction(
    endpoint="/api/v1/widgets/github/stats",
    headers={
        "Accept": "application/vnd.github.v3+json",
        "Authorization": "token ghp_xxxxx"
    },
    refresh_interval=1800,  # Refresh every 30 minutes
    func=process_github_data
)

await widget.save()
```

### Interaction Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `endpoint` | `str` | Widget endpoint path (e.g., `/api/v1/widgets/mywidget/action`) |
| `headers` | `Dict[str, str]` | HTTP headers for the interaction |
| `refresh_interval` | `int` | Auto-refresh interval in seconds (0 = no refresh) |
| `func` | `Callable` | Python function to process widget data |

## Adding Components to Widgets

### Component Structure

Components define the UI elements of your widget. Each component has:

- **`type`** - Component type (e.g., "button", "input", "text", "container")
- **`content`** - Data stored in the component (list of any type)
- **`props`** - Component properties (styling, positioning, behavior)

### Creating Components

```python
# Create widget first
await widget.create(
    name="Contact Form",
    size=WidgetSize.MEDIUM.value
)

# Define interaction
await widget.interaction(
    endpoint="/api/v1/widgets/contact/submit",
    headers={"Content-Type": "application/json"},
    refresh_interval=0,
    func=handle_contact_submission
)

# Add a text input component
await widget.component(
    endpoint="/api/v1/widgets/contact/submit",
    type="input",
    content=[],
    props={
        "id": "name_input",
        "name": "user_name",
        "type": "text",
        "placeholder": "Enter your name",
        "width": "100%",
        "height": 40,
        "border": "1px solid #ccc",
        "border_radius": 4,
        "padding": {"top": 8, "right": 12, "bottom": 8, "left": 12}
    }
)

# Add a submit button
await widget.component(
    endpoint="/api/v1/widgets/contact/submit",
    type="button",
    content=[],
    props={
        "id": "submit_button",
        "label": "Submit",
        "width": 100,
        "height": 40,
        "background": "#007bff",
        "color": "#ffffff",
        "border": "none",
        "border_radius": 4,
        "cursor": "pointer",
        "on_click": "submit_form"
    }
)

await widget.save()
```

### Component Types

| Type | Description | Common Props |
|------|-------------|-------------|
| `container` | Layout container for other components | `display`, `flex_direction`, `padding`, `background` |
| `text` | Text display | `content`, `font_size`, `font_weight`, `color` |
| `input` | Text input field | `placeholder`, `max_length`, `type`, `name` |
| `button` | Clickable button | `label`, `on_click`, `background`, `color` |
| `badge` | Info badge/label | `label`, `background`, `color`, `border_radius` |
| `message_list` | Scrollable message list | `overflow_y`, `render_type`, `item_template` |
| `message_item` | Message item template | `is_template`, `username`, `text`, `timestamp` |

### Component Properties (Props)

All components support flexible properties through the `props` dictionary:

```python
props = {
    # Positioning
    "position": "relative",  # or "absolute"
    "top": 0,
    "left": 0,
    "bottom": 0,
    "right": 0,
    
    # Sizing
    "width": "100%",
    "height": 50,
    "flex": 1,
    
    # Styling
    "background": "#ffffff",
    "color": "#000000",
    "border": "1px solid #ccc",
    "border_radius": 8,
    "padding": {"top": 12, "right": 16, "bottom": 12, "left": 16},
    "margin": 10,
    
    # Layout
    "display": "flex",
    "flex_direction": "column",
    "align_items": "center",
    "justify_content": "space-between",
    "gap": 12,
    
    # Typography
    "font_size": 14,
    "font_weight": "bold",
    "text_align": "center",
    "line_height": 1.5,
    
    # Behavior
    "cursor": "pointer",
    "overflow_y": "auto",
    "on_click": "action_name",
    "on_enter": "submit_action"
}
```

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

## Advanced Widget Patterns

### Reusable Widget Factory

```python
from lazi_sdk import WriteWidget
from api.schemas.widgets import WidgetSize

class WidgetFactory:
    """Factory for creating standard widget types"""
    
    def __init__(self, username: str, token: str, project_id: str):
        self.username = username
        self.token = token
        self.project_id = project_id
    
    async def create_analytics_widget(self, name: str, endpoint: str):
        """Create a standard analytics widget"""
        widget = WriteWidget(
            username=self.username,
            token=self.token,
            project_id=self.project_id
        )
        
        await widget.create(
            name=name,
            size=WidgetSize.LARGE.value
        )
        
        def analytics_handler(data, req, user):
            return {
                'metrics': data.get('metrics', {}),
                'updated': data.get('timestamp')
            }
        
        await widget.interaction(
            endpoint=endpoint,
            headers={"Content-Type": "application/json"},
            refresh_interval=300,
            func=analytics_handler
        )
        
        await widget.save()
        return widget.current_widget

# Usage
factory = WidgetFactory(
    username="user@example.com",
    token=token,
    project_id="proj_001"
)

widget = await factory.create_analytics_widget(
    name="Sales Analytics",
    endpoint="/api/v1/widgets/sales/metrics"
)
```

## Widget Updates

### Modify Existing Widget

```python
from lazi_sdk import ReadWidget, WriteWidget
from api.schemas.widgets import WidgetConfig, WidgetSize

# Step 1: Get existing widget
reader = ReadWidget(
    username="user@example.com",
    token=token,
    project_id="project_123"
)
widget_data = await reader.get_widget("widget_123")

# Step 2: Initialize writer with existing widget
writer = WriteWidget(
    username="user@example.com",
    token=token,
    project_id="project_123"
)

# Load existing widget configuration
writer.current_widget = WidgetConfig(**widget_data)

# Step 3: Update properties
writer.current_widget.name = "Updated Dashboard"
writer.current_widget.size = WidgetSize.EXTRA_LARGE

# Step 4: Add new interaction
def new_handler(data, req, user):
    return {'status': 'updated', 'data': data}

await writer.interaction(
    endpoint="/api/v1/widgets/updated/endpoint",
    headers={"Content-Type": "application/json"},
    refresh_interval=120,
    func=new_handler
)

# Step 5: Save changes
await writer.save()
print(f"Widget updated: {writer.current_widget.id}")
```

## Complete Example

### Building a Weather Widget

```python
from lazi_sdk import WriteWidget
from api.schemas.widgets import WidgetSize
import os

async def create_weather_widget():
    # Initialize
    widget = WriteWidget(
        username=os.getenv("LAZI_USERNAME"),
        token=os.getenv("LAZI_TOKEN"),
        project_id="weather_project_001"
    )
    
    # Create widget
    await widget.create(
        name="Weather Dashboard",
        size=WidgetSize.MEDIUM.value
    )
    
    # Define weather processing function
    def process_weather_data(widget_data, request, user):
        temp = widget_data.get('main', {}).get('temp', 0)
        feels_like = widget_data.get('main', {}).get('feels_like', 0)
        humidity = widget_data.get('main', {}).get('humidity', 0)
        description = widget_data.get('weather', [{}])[0].get('description', 'N/A')
        
        return {
            'temperature': f"{temp}°C",
            'feels_like': f"{feels_like}°C",
            'humidity': f"{humidity}%",
            'conditions': description.title(),
            'last_updated': widget_data.get('dt')
        }
    
    # Configure weather API interaction
    await widget.interaction(
        endpoint="/api/v1/widgets/weather/update",
        headers={"Content-Type": "application/json"},
        refresh_interval=600,  # Refresh every 10 minutes
        func=process_weather_data
    )
    
    # Add temperature display component
    await widget.component(
        endpoint="/api/v1/widgets/weather/update",
        type="text",
        content=[],
        props={
            "id": "temperature_display",
            "content": "{{temperature}}",
            "font_size": 48,
            "font_weight": "bold",
            "color": "#333",
            "text_align": "center"
        }
    )
    
    # Add conditions text
    await widget.component(
        endpoint="/api/v1/widgets/weather/update",
        type="text",
        content=[],
        props={
            "id": "conditions_text",
            "content": "{{conditions}}",
            "font_size": 18,
            "color": "#666",
            "text_align": "center",
            "margin_top": 8
        }
    )
    
    # Upload weather icon using S3
    presigned_url = await widget.uploadable(
        object_name="weather_icon.png",
        filename="./assets/weather_icon.png",
        expire=3600
    )
    
    # Save
    await widget.save()
    
    print(f"✅ Weather widget created: {widget.current_widget.id}")
    print(f"   Icon URL: {presigned_url}")
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

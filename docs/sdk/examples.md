# SDK Examples

Complete, real-world examples using the Lazi SDK.

## Table of Contents

- [Complete OAuth2 + Widget Creation](#complete-oauth2--widget-creation)
- [Discussion Board Widget via Public API](#discussion-board-widget-via-public-api)
- [Basic Widget Creation](#basic-widget-creation)
- [Analytics Dashboard](#analytics-dashboard)
- [GitHub Integration](#github-integration)
- [Multi-Widget Project](#multi-widget-project)
- [Real-Time Data Widget](#real-time-data-widget)
- [Image Gallery Widget](#image-gallery-widget)

## Complete OAuth2 + Widget Creation

**End-to-end example showing OAuth2 authentication and widget creation:**

```python
import asyncio
import requests
import os
from lazi_sdk import WriteWidget
from dotenv import load_dotenv

load_dotenv()

class WidgetCreationWorkflow:
    """Complete workflow: OAuth2 authentication → Widget creation → Save"""
    
    def __init__(self):
        self.base_url = os.getenv("LAZI_API_URL", "https://api.lazi.com")
        self.access_token = None
        self.refresh_token = None
        self.user_id = None
    
    def step_1_authenticate(self, username: str, password: str):
        """
        Step 1: Authenticate and get OAuth2 tokens
        """
        print("\n🔐 Step 1: Authenticating with OAuth2...")
        
        response = requests.post(
            f"{self.base_url}/oauth/token",
            data={
                "grant_type": "password",
                "username": username,
                "password": password,
                "scope": "widgets:write widgets:read projects:read files:write"
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        if response.status_code != 200:
            raise Exception(f"Authentication failed: {response.text}")
        
        tokens = response.json()
        self.access_token = tokens["access_token"]
        self.refresh_token = tokens["refresh_token"]
        self.user_id = tokens["user_id"]  # Assuming API returns user_id
        
        print(f"✅ Authenticated successfully")
        print(f"   Token type: {tokens['token_type']}")
        print(f"   Expires in: {tokens['expires_in']} seconds")
        print(f"   Scopes: {tokens['scope']}")
        
        return tokens
    
    async def step_2_create_widget(self, project_id: str):
        """
        Step 2: Create widget using SDK
        """
        print("\n🎨 Step 2: Creating widget with SDK...")
        
        # Initialize SDK with OAuth token
        widget = WriteWidget(
            token=self.access_token,
            user_id=self.user_id
        )
        
        # Create widget configuration
        await widget.create_new(
            name="Sales Dashboard",
            description="Real-time sales metrics",
            size="large",
            project_id=project_id
        )
        
        print(f"✅ Widget created")
        print(f"   Widget ID: {widget.current_widget.id}")
        print(f"   Name: {widget.current_widget.name}")
        print(f"   Size: {widget.current_widget.size}")
        
        return widget
    
    async def step_3_add_interaction(self, widget: WriteWidget):
        """
        Step 3: Add interaction endpoint and logic
        """
        print("\n⚙️  Step 3: Adding interaction endpoint...")
        
        await widget.interaction(
            endpoint="/api/v1/widgets/sales/update",
            headers={"Content-Type": "application/json"},
            refresh_interval=300,  # Refresh every 5 minutes
            func="""
def process_data(widget_data, request, user):
    # Process sales data
    revenue = widget_data.get('revenue', 0)
    transactions = widget_data.get('transactions', [])
    
    return {
        'total_revenue': f"${revenue:,.2f}",
        'transaction_count': len(transactions),
        'avg_transaction': f"${revenue/len(transactions) if transactions else 0:.2f}",
        'status': 'healthy' if revenue > 10000 else 'needs_attention'
    }
            """
        )
        
        print(f"✅ Interaction added")
        print(f"   Endpoint: /api/v1/widgets/sales/update")
        print(f"   Refresh interval: 300 seconds")
        
        return widget
    
    async def step_4_save_widget(self, widget: WriteWidget):
        """
        Step 4: Save widget to MongoDB
        """
        print("\n💾 Step 4: Saving widget to database...")
        
        await widget.save()
        
        print(f"✅ Widget saved successfully")
        print(f"   Widget ID: {widget.current_widget.id}")
        print(f"   MongoDB collection: openWidgets")
        
        return widget.current_widget.id
    
    async def run_complete_flow(self, username: str, password: str, project_id: str):
        """
        Execute complete OAuth2 → Widget creation workflow
        """
        print("="*60)
        print("🚀 Starting Complete Widget Creation Workflow")
        print("="*60)
        
        # Step 1: Authenticate
        self.step_1_authenticate(username, password)
        
        # Step 2: Create widget
        widget = await self.step_2_create_widget(project_id)
        
        # Step 3: Add interaction
        widget = await self.step_3_add_interaction(widget)
        
        # Step 4: Save widget
        widget_id = await self.step_4_save_widget(widget)
        
        print("\n" + "="*60)
        print("✅ Workflow completed successfully!")
        print("="*60)
        print(f"\n📋 Summary:")
        print(f"   Widget ID: {widget_id}")
        print(f"   Access Token: {self.access_token[:20]}...")
        print(f"   User ID: {self.user_id}")
        print(f"\n🔗 Next Steps:")
        print(f"   - Use widget ID to interact via public API")
        print(f"   - Refresh token before it expires (in 1 hour)")
        print(f"   - Test widget endpoint: /api/v1/public/widgets/{widget_id}")
        
        return widget_id


# Run the complete workflow
if __name__ == "__main__":
    workflow = WidgetCreationWorkflow()
    
    widget_id = asyncio.run(workflow.run_complete_flow(
        username=os.getenv("LAZI_USERNAME"),
        password=os.getenv("LAZI_PASSWORD"),
        project_id=os.getenv("PROJECT_ID")
    ))
    
    print(f"\n🎉 Widget ready to use: {widget_id}")
```

**Output:**

```
============================================================
🚀 Starting Complete Widget Creation Workflow
============================================================

🔐 Step 1: Authenticating with OAuth2...
✅ Authenticated successfully
   Token type: Bearer
   Expires in: 3600 seconds
   Scopes: widgets:write widgets:read projects:read files:write

🎨 Step 2: Creating widget with SDK...
✅ Widget created
   Widget ID: wgt_abc123xyz789
   Name: Sales Dashboard
   Size: large

⚙️  Step 3: Adding interaction endpoint...
✅ Interaction added
   Endpoint: /api/v1/widgets/sales/update
   Refresh interval: 300 seconds

💾 Step 4: Saving widget to database...
✅ Widget saved successfully
   Widget ID: wgt_abc123xyz789
   MongoDB collection: openWidgets

============================================================
✅ Workflow completed successfully!
============================================================

📋 Summary:
   Widget ID: wgt_abc123xyz789
   Access Token: eyJhbGciOiJIUzI1NiIs...
   User ID: usr_456def

🔗 Next Steps:
   - Use widget ID to interact via public API
   - Refresh token before it expires (in 1 hour)
   - Test widget endpoint: /api/v1/public/widgets/wgt_abc123xyz789

🎉 Widget ready to use: wgt_abc123xyz789
```

## Discussion Board Widget via Public API

**Create a chat/discussion widget that stores messages using the public API:**

```python
import asyncio
import os
from lazi_sdk import WriteWidget
from api.schemas.widgets import WidgetComponents

class DiscussionWidget:
    """Create a discussion board widget using public API"""
    
    def __init__(self, token: str, user_id: str):
        self.w_client = WriteWidget(token=token, user_id=user_id)
        self.widget_id = None
    
    async def create_discussion_board(self, project_id: str, board_name: str):
        """
        Create complete discussion board widget
        
        Features:
        - Real-time message posting
        - User avatars and timestamps
        - Message reactions
        - Auto-refresh every 10 seconds
        - WebSocket support for instant updates
        """
        print(f"\n📋 Creating Discussion Board: {board_name}")
        
        # Step 1: Create widget
        await self.w_client.create_new(
            name=board_name,
            description="Team discussion and chat",
            size="extra_large",
            project_id=project_id
        )
        
        # Step 2: Define UI components
        components = [
            # Header
            WidgetComponents(
                type="header",
                content=["Discussion Board"],
                properties={
                    "position": {"x": 0, "y": 0},
                    "padding": {"top": 16, "bottom": 16, "left": 20, "right": 20},
                    "styling": {
                        "backgroundColor": "#1a1a2e",
                        "color": "#ffffff",
                        "fontSize": 24,
                        "fontWeight": "bold"
                    }
                }
            ),
            # Message list container - stores messages in content field
            WidgetComponents(
                type="message_list",
                content=[],  # Messages stored here as dictionaries
                properties={
                    "position": {"x": 0, "y": 60},
                    "dimensions": {"width": "100%", "height": 500},
                    "styling": {
                        "overflowY": "scroll",
                        "padding": 20,
                        "backgroundColor": "#f5f5f5"
                    }
                }
            ),
            # Input field
            WidgetComponents(
                type="text_input",
                content=[],
                properties={
                    "position": {"x": 20, "y": 580},
                    "dimensions": {"width": "calc(100% - 140px)", "height": 50},
                    "placeholder": "Type your message...",
                    "styling": {
                        "fontSize": 16,
                        "padding": 12,
                        "borderRadius": 8
                    }
                }
            ),
            # Send button
            WidgetComponents(
                type="button",
                content=["Send"],
                properties={
                    "position": {"x": "calc(100% - 100px)", "y": 580},
                    "dimensions": {"width": 80, "height": 50},
                    "action": "post_message",
                    "styling": {
                        "backgroundColor": "#4CAF50",
                        "color": "#ffffff",
                        "borderRadius": 8,
                        "fontWeight": "bold"
                    }
                }
            )
        ]
        
        # Step 3: Add all components
        for component in components:
            await self.w_client.component(
                type=component.type,
                content=component.content,
                properties=component.properties
            )
        
        # Step 4: Define message posting logic
        message_handler = """
def process_data(widget_data, request, user):
    import uuid
    from datetime import datetime
    
    # Get message text from request
    message_text = request.get('message_text', '').strip()
    
    if not message_text:
        return {'status': 'error', 'message': 'Message cannot be empty'}
    
    # Retrieve existing messages from component content field
    existing_messages = widget_data.get('component_content', {}).get('message_list', [])
    
    # Create new message dictionary
    new_message = {
        'id': str(uuid.uuid4()),
        'user_id': user.get('user_id'),
        'username': user.get('username', 'Anonymous'),
        'text': message_text,
        'timestamp': datetime.utcnow().isoformat(),
        'edited': False,
        'reactions': []
    }
    
    # Add to messages list
    existing_messages.append(new_message)
    
    # Return component updates to store messages in content field
    return {
        'status': 'success',
        'message': 'Message posted',
        'component_updates': {
            'message_list': {
                'content': existing_messages  # Update content field with all messages
            }
        },
        'data': {
            'messages': existing_messages[-50:],  # Display last 50 messages
            'total_messages': len(existing_messages),
            'new_message': new_message
        },
        'broadcast': True,  # Broadcast to all connected clients via WebSocket
        'broadcast_event': 'message_posted',
        'broadcast_data': {
            'message': new_message,
            'total_count': len(existing_messages)
        }
    }
        """
        
        # Step 5: Configure interaction endpoint
        await self.w_client.interaction(
            endpoint="/api/v1/widgets/discussion/post_message",
            headers={
                "Content-Type": "application/json",
                "X-WebSocket-Enabled": "true"  # Enable WebSocket for real-time
            },
            refresh_interval=10,  # Fallback: Refresh every 10 seconds
            func=message_handler
        )
        
        # Step 6: Save widget
        await self.w_client.save()
        self.widget_id = self.w_client.current_widget.id
        
        print(f"✅ Discussion board created successfully!")
        print(f"   Widget ID: {self.widget_id}")
        print(f"   Components: {len(components)}")
        print(f"   Endpoint: /api/v1/widgets/discussion/post_message")
        print(f"   WebSocket: Enabled for real-time updates")
        
        return self.widget_id


# Usage
async def main():
    # Authenticate first (see OAuth2 example above)
    discussion = DiscussionWidget(
        token=os.getenv("LAZI_TOKEN"),
        user_id=os.getenv("LAZI_USER_ID")
    )
    
    widget_id = await discussion.create_discussion_board(
        project_id=os.getenv("PROJECT_ID"),
        board_name="Team Chat"
    )
    
    print(f"\n🎉 Ready to chat! Widget ID: {widget_id}")

if __name__ == "__main__":
    asyncio.run(main())
```

**How Messages Are Stored:**

```json
{
  "widget_id": "wgt_discussion_123",
  "interactions": {
    "/api/v1/widgets/discussion/post_message": {
      "logic": "def process_data...",
      "components": [
        {
          "type": "message_list",
          "content": [
            {
              "id": "msg_001",
              "user_id": "usr_456",
              "username": "Alice",
              "text": "Hello team!",
              "timestamp": "2025-11-30T10:30:00Z",
              "edited": false,
              "reactions": ["👍", "🎉"]
            },
            {
              "id": "msg_002",
              "user_id": "usr_789",
              "username": "Bob",
              "text": "Hi Alice! Great to see you.",
              "timestamp": "2025-11-30T10:31:15Z",
              "edited": false,
              "reactions": []
            }
          ],
          "properties": {...}
        }
      ]
    }
  }
}
```

## Basic Widget Creation

Simple widget creation and management:

```python
import asyncio
import os
from lazi_sdk import WriteWidget
from dotenv import load_dotenv

load_dotenv()

async def create_simple_widget():
    """Create a basic widget"""
    
    # Initialize SDK
    widget = WriteWidget(
        token=os.getenv("LAZI_TOKEN"),
        user_id=os.getenv("LAZI_USER_ID")
    )
    
    # Create widget
    await widget.create_new(
        name="My First Widget",
        description="A simple example widget",
        size="medium",
        project_id="demo_project_001"
    )
    
    # Save
    await widget.save()
    
    print(f"✅ Widget created: {widget.current_widget.id}")
    print(f"   Name: {widget.current_widget.name}")
    print(f"   Size: {widget.current_widget.size}")

if __name__ == "__main__":
    asyncio.run(create_simple_widget())
```

## Analytics Dashboard

Create a dashboard widget that fetches real-time analytics:

```python
import asyncio
import os
from lazi_sdk import WriteWidget

async def create_analytics_dashboard():
    """Create analytics dashboard widget"""
    
    widget = WriteWidget(
        token=os.getenv("LAZI_TOKEN"),
        user_id=os.getenv("LAZI_USER_ID")
    )
    
    # Create widget
    await widget.create_new(
        name="Sales Analytics Dashboard",
        description="Real-time sales metrics and KPIs",
        size="extra_large",
        project_id=os.getenv("PROJECT_ID")
    )
    
    # Configure analytics endpoint
    await widget.post(
        endpoint="https://analytics.example.com/api/v1/dashboard",
        endpoint_data={
            "params": {
                "metrics": "revenue,users,conversion",
                "date_range": "last_7_days",
                "granularity": "daily"
            },
            "headers": {
                "Authorization": f"Bearer {os.getenv('ANALYTICS_TOKEN')}",
                "Content-Type": "application/json"
            },
            "refresh_interval": 300,  # Update every 5 minutes
            "logic": """
def process_data(widget_data, request, user):
    # Extract key metrics
    revenue = widget_data.get('revenue', {})
    users = widget_data.get('users', {})
    conversion = widget_data.get('conversion', {})
    
    # Calculate trends
    revenue_trend = revenue.get('current', 0) - revenue.get('previous', 0)
    user_growth = ((users.get('current', 0) - users.get('previous', 0)) / users.get('previous', 1)) * 100
    
    return {
        'kpis': {
            'revenue': {
                'value': f"${revenue.get('current', 0):,.2f}",
                'trend': 'up' if revenue_trend > 0 else 'down',
                'change': f"{abs(revenue_trend):,.2f}"
            },
            'users': {
                'value': users.get('current', 0),
                'growth': f"{user_growth:.1f}%"
            },
            'conversion_rate': {
                'value': f"{conversion.get('rate', 0):.2f}%",
                'status': 'good' if conversion.get('rate', 0) > 2.5 else 'needs_improvement'
            }
        },
        'charts': {
            'revenue_over_time': widget_data.get('revenue_timeline', []),
            'user_acquisition': widget_data.get('user_timeline', [])
        },
        'updated_at': widget_data.get('timestamp')
    }
            """
        }
    )
    
    # Attach dashboard logo
    await widget.attach_media(
        object_name="dashboard_logo.png",
        filename="./assets/company_logo.png"
    )
    
    await widget.save()
    print(f"✅ Analytics dashboard created: {widget.current_widget.id}")

asyncio.run(create_analytics_dashboard())
```

## GitHub Integration

Widget that displays GitHub repository statistics:

```python
import asyncio
import os
from lazi_sdk import WriteWidget

async def create_github_widget():
    """Create GitHub repository stats widget"""
    
    widget = WriteWidget(
        token=os.getenv("LAZI_TOKEN"),
        user_id=os.getenv("LAZI_USER_ID")
    )
    
    # GitHub repository details
    repo_owner = "Jacey1225"
    repo_name = "personal-manager-02"
    
    await widget.create_new(
        name=f"{repo_owner}/{repo_name}",
        description="GitHub repository statistics",
        size="large",
        project_id=os.getenv("PROJECT_ID")
    )
    
    await widget.post(
        endpoint=f"https://api.github.com/repos/{repo_owner}/{repo_name}",
        endpoint_data={
            "params": {},
            "headers": {
                "Accept": "application/vnd.github.v3+json",
                "Authorization": f"token {os.getenv('GITHUB_TOKEN')}"
            },
            "refresh_interval": 3600,  # Update hourly
            "logic": """
def process_data(widget_data, request, user):
    return {
        'repository': {
            'name': widget_data.get('full_name'),
            'description': widget_data.get('description'),
            'url': widget_data.get('html_url')
        },
        'stats': {
            'stars': widget_data.get('stargazers_count', 0),
            'forks': widget_data.get('forks_count', 0),
            'watchers': widget_data.get('watchers_count', 0),
            'open_issues': widget_data.get('open_issues_count', 0)
        },
        'metadata': {
            'language': widget_data.get('language'),
            'license': widget_data.get('license', {}).get('name', 'None'),
            'created_at': widget_data.get('created_at'),
            'updated_at': widget_data.get('updated_at')
        },
        'popularity': 'high' if widget_data.get('stargazers_count', 0) > 100 else 'moderate'
    }
            """
        }
    )
    
    await widget.save()
    print(f"✅ GitHub widget created: {widget.current_widget.id}")

asyncio.run(create_github_widget())
```

## Multi-Widget Project

Create multiple widgets for a single project:

```python
import asyncio
import os
from lazi_sdk import WriteWidget, ReadWidget

async def create_project_dashboard():
    """Create a complete dashboard with multiple widgets"""
    
    project_id = os.getenv("PROJECT_ID")
    token = os.getenv("LAZI_TOKEN")
    user_id = os.getenv("LAZI_USER_ID")
    
    widgets_config = [
        {
            "name": "Team Overview",
            "description": "Team members and roles",
            "size": "medium",
            "endpoint": "/api/v1/team/overview",
            "refresh": 0
        },
        {
            "name": "Recent Activity",
            "description": "Latest project updates",
            "size": "medium",
            "endpoint": "/api/v1/activity/recent",
            "refresh": 60
        },
        {
            "name": "Task Progress",
            "description": "Current sprint progress",
            "size": "large",
            "endpoint": "/api/v1/tasks/progress",
            "refresh": 300
        },
        {
            "name": "Quick Stats",
            "description": "Key project metrics",
            "size": "small",
            "endpoint": "/api/v1/stats/quick",
            "refresh": 180
        }
    ]
    
    created_widgets = []
    
    for config in widgets_config:
        widget = WriteWidget(token=token, user_id=user_id)
        
        await widget.create_new(
            name=config["name"],
            description=config["description"],
            size=config["size"],
            project_id=project_id
        )
        
        await widget.post(
            endpoint=config["endpoint"],
            endpoint_data={
                "params": {"project_id": project_id},
                "headers": {"Authorization": f"Bearer {token}"},
                "refresh_interval": config["refresh"],
                "logic": "def process_data(data, req, user): return data"
            }
        )
        
        await widget.save()
        created_widgets.append(widget.current_widget.id)
        print(f"✅ Created: {config['name']}")
    
    # Verify all widgets
    reader = ReadWidget(
        username=os.getenv("LAZI_USERNAME"),
        token=token,
        project_id=project_id
    )
    
    all_widgets = await reader.list_widgets()
    print(f"\n📊 Total widgets in project: {len(all_widgets)}")
    
    return created_widgets

asyncio.run(create_project_dashboard())
```

## Real-Time Data Widget

Widget with real-time stock market data:

```python
import asyncio
import os
from lazi_sdk import WriteWidget

async def create_stock_widget():
    """Create real-time stock ticker widget"""
    
    widget = WriteWidget(
        token=os.getenv("LAZI_TOKEN"),
        user_id=os.getenv("LAZI_USER_ID")
    )
    
    await widget.create_new(
        name="Stock Market Ticker",
        description="Real-time stock prices",
        size="medium",
        project_id=os.getenv("PROJECT_ID")
    )
    
    await widget.post(
        endpoint="https://api.marketdata.app/v1/stocks/quotes/AAPL,GOOGL,MSFT",
        endpoint_data={
            "params": {
                "format": "json"
            },
            "headers": {
                "Authorization": f"Bearer {os.getenv('MARKET_DATA_TOKEN')}"
            },
            "refresh_interval": 60,  # Update every minute
            "logic": """
def process_data(widget_data, request, user):
    stocks = []
    
    for quote in widget_data.get('quotes', []):
        symbol = quote.get('symbol')
        price = quote.get('last', 0)
        change = quote.get('change', 0)
        change_pct = quote.get('changepct', 0)
        
        stocks.append({
            'symbol': symbol,
            'price': f"${price:.2f}",
            'change': f"${change:.2f}",
            'change_percent': f"{change_pct:+.2f}%",
            'trend': 'up' if change > 0 else 'down' if change < 0 else 'flat',
            'volume': quote.get('volume', 0)
        })
    
    return {
        'stocks': stocks,
        'market_status': 'open' if widget_data.get('market_open') else 'closed',
        'last_updated': widget_data.get('timestamp')
    }
            """
        }
    )
    
    await widget.save()
    print(f"✅ Stock ticker created: {widget.current_widget.id}")

asyncio.run(create_stock_widget())
```

## Image Gallery Widget

Widget displaying an image gallery with media files:

```python
import asyncio
import os
from pathlib import Path
from lazi_sdk import WriteWidget

async def create_gallery_widget():
    """Create image gallery widget"""
    
    widget = WriteWidget(
        token=os.getenv("LAZI_TOKEN"),
        user_id=os.getenv("LAZI_USER_ID")
    )
    
    await widget.create_new(
        name="Product Gallery",
        description="Product images and media",
        size="extra_large",
        project_id=os.getenv("PROJECT_ID")
    )
    
    # Upload multiple images
    image_dir = Path("./product_images")
    
    if image_dir.exists():
        for image_file in image_dir.glob("*.{jpg,png,jpeg}"):
            print(f"Uploading {image_file.name}...")
            await widget.attach_media(
                object_name=image_file.name,
                filename=str(image_file)
            )
    
    # Configure gallery interaction
    await widget.post(
        endpoint="/api/v1/products/gallery",
        endpoint_data={
            "params": {
                "category": "electronics",
                "limit": 20
            },
            "headers": {},
            "refresh_interval": 0,  # Static gallery
            "logic": """
def process_data(widget_data, request, user):
    # Organize images into categories
    images = widget_data.get('images', [])
    
    return {
        'gallery': {
            'total_images': len(images),
            'categories': {
                'featured': [img for img in images if img.get('featured')],
                'new': [img for img in images if img.get('is_new')],
                'all': images
            }
        },
        'layout': 'grid',
        'thumbnail_size': 'medium'
    }
            """
        }
    )
    
    await widget.save()
    print(f"✅ Gallery widget created: {widget.current_widget.id}")
    print(f"   Total media files: {len(widget.current_widget.content)}")

asyncio.run(create_gallery_widget())
```

## Testing and Error Handling

Complete example with error handling:

```python
import asyncio
import os
from lazi_sdk import WriteWidget
from fastapi import HTTPException

async def robust_widget_creation():
    """Widget creation with comprehensive error handling"""
    
    try:
        # Validate environment
        required_vars = ["LAZI_TOKEN", "LAZI_USER_ID", "PROJECT_ID"]
        for var in required_vars:
            if not os.getenv(var):
                raise ValueError(f"Missing environment variable: {var}")
        
        # Initialize widget
        widget = WriteWidget(
            token=os.getenv("LAZI_TOKEN"),
            user_id=os.getenv("LAZI_USER_ID")
        )
        
        # Create widget
        await widget.create_new(
            name="Robust Widget",
            description="Widget with error handling",
            size="medium",
            project_id=os.getenv("PROJECT_ID")
        )
        
        # Save
        await widget.save()
        
        print(f"✅ Success! Widget ID: {widget.current_widget.id}")
        return widget.current_widget.id
        
    except ValueError as e:
        print(f"❌ Configuration error: {str(e)}")
        return None
        
    except HTTPException as e:
        if e.status_code == 401:
            print("❌ Authentication failed: Check your token")
        elif e.status_code == 403:
            print("❌ Permission denied: Insufficient scopes")
        elif e.status_code == 404:
            print("❌ Not found: Check your project ID")
        else:
            print(f"❌ HTTP error: {e.detail}")
        return None
        
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return None

asyncio.run(robust_widget_creation())
```

## Next Steps

- [Authentication Guide](authentication.md) - Learn more about OAuth2 authentication
- [Widget Management](widgets.md) - Detailed widget API documentation
- [Custom Endpoints](../advanced/custom-endpoints.md) - Create custom widget endpoints

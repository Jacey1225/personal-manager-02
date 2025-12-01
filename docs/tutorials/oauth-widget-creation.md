# Complete Tutorial: OAuth2 Authentication & Widget Creation

This tutorial walks you through the complete process of authenticating with OAuth2 and creating a widget using the public API.

## Prerequisites

- Python 3.8+
- Lazi SDK installed (`pip install lazi-sdk`)
- Lazi account with developer access
- Project ID from your Lazi dashboard

## What You'll Build

By the end of this tutorial, you'll have:

1. ✅ OAuth2 authentication working
2. ✅ Access and refresh token management
3. ✅ A functional widget created via the SDK
4. ✅ Custom interaction endpoint with logic
5. ✅ Widget saved to MongoDB

---

## Part 1: OAuth2 Authentication

### Step 1: Set Up Environment

Create a `.env` file in your project root:

```env
LAZI_API_URL=https://api.lazi.com
LAZI_USERNAME=your_email@example.com
LAZI_PASSWORD=your_secure_password
PROJECT_ID=your_project_id
```

Install dependencies:

```bash
pip install requests python-dotenv lazi-sdk
```

### Step 2: Authenticate and Get Tokens

Create `authenticate.py`:

```python
import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

def authenticate():
    """Get OAuth2 access token"""
    
    # API endpoint
    token_url = f"{os.getenv('LAZI_API_URL')}/oauth/token"
    
    # Authentication data
    auth_data = {
        "grant_type": "password",
        "username": os.getenv("LAZI_USERNAME"),
        "password": os.getenv("LAZI_PASSWORD"),
        "scope": "widgets:write widgets:read projects:read files:write"
    }
    
    # Make request
    response = requests.post(
        token_url,
        data=auth_data,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    
    # Check response
    if response.status_code == 200:
        tokens = response.json()
        
        print("✅ Authentication successful!")
        print(f"   Access Token: {tokens['access_token'][:30]}...")
        print(f"   Refresh Token: {tokens['refresh_token'][:30]}...")
        print(f"   Token Type: {tokens['token_type']}")
        print(f"   Expires In: {tokens['expires_in']} seconds (1 hour)")
        print(f"   Scopes: {tokens['scope']}")
        
        # Save tokens to .env for later use
        with open('.env', 'a') as f:
            f.write(f"\nLAZI_ACCESS_TOKEN={tokens['access_token']}")
            f.write(f"\nLAZI_REFRESH_TOKEN={tokens['refresh_token']}")
        
        return tokens
    else:
        print(f"❌ Authentication failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return None

if __name__ == "__main__":
    authenticate()
```

**Run it:**

```bash
python authenticate.py
```

**Expected Output:**

```
✅ Authentication successful!
   Access Token: eyJhbGciOiJIUzI1NiIsInR5cCI6Ik...
   Refresh Token: eyJhbGciOiJIUzI1NiIsInR5cCI6Ik...
   Token Type: Bearer
   Expires In: 3600 seconds (1 hour)
   Scopes: widgets:write widgets:read projects:read files:write
```

### Step 3: Implement Token Refresh

Create `refresh_token.py`:

```python
import requests
import os
from dotenv import load_dotenv

load_dotenv()

def refresh_access_token():
    """Refresh expired access token"""
    
    refresh_url = f"{os.getenv('LAZI_API_URL')}/oauth/token/refresh"
    
    response = requests.post(
        refresh_url,
        json={"refresh_token": os.getenv("LAZI_REFRESH_TOKEN")},
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        tokens = response.json()
        print("✅ Token refreshed successfully!")
        print(f"   New Access Token: {tokens['access_token'][:30]}...")
        
        # Update .env file
        # (In production, use a proper config manager)
        
        return tokens
    else:
        print(f"❌ Token refresh failed: {response.status_code}")
        return None

if __name__ == "__main__":
    refresh_access_token()
```

---

## Part 2: Create a Widget

### Step 4: Initialize SDK

Create `create_widget.py`:

```python
import asyncio
import os
from dotenv import load_dotenv
from lazi_sdk import WriteWidget

load_dotenv()

async def create_simple_widget():
    """Create a basic widget using the SDK"""
    
    print("\n🎨 Creating widget...\n")
    
    # Initialize SDK with OAuth token
    widget = WriteWidget(
        token=os.getenv("LAZI_ACCESS_TOKEN"),
        user_id=os.getenv("LAZI_USER_ID", "default_user")
    )
    
    # Create widget configuration
    await widget.create_new(
        name="My First Widget",
        description="Created via OAuth2 and SDK",
        size="medium",  # Options: small, medium, large, extra_large
        project_id=os.getenv("PROJECT_ID")
    )
    
    print("✅ Widget created!")
    print(f"   Widget ID: {widget.current_widget.id}")
    print(f"   Name: {widget.current_widget.name}")
    print(f"   Size: {widget.current_widget.size}")
    
    return widget

if __name__ == "__main__":
    asyncio.run(create_simple_widget())
```

**Run it:**

```bash
python create_widget.py
```

**Expected Output:**

```
🎨 Creating widget...

✅ Widget created!
   Widget ID: wgt_a1b2c3d4e5f6
   Name: My First Widget
   Size: medium
```

### Step 5: Add Interaction Endpoint

Enhance `create_widget.py`:

```python
import asyncio
import os
from dotenv import load_dotenv
from lazi_sdk import WriteWidget

load_dotenv()

async def create_widget_with_interaction():
    """Create widget with custom interaction logic"""
    
    print("\n🎨 Creating widget with interaction...\n")
    
    # Initialize
    widget = WriteWidget(
        token=os.getenv("LAZI_ACCESS_TOKEN"),
        user_id=os.getenv("LAZI_USER_ID", "default_user")
    )
    
    # Step 1: Create widget
    await widget.create_new(
        name="Weather Widget",
        description="Shows current weather",
        size="medium",
        project_id=os.getenv("PROJECT_ID")
    )
    
    print("✅ Widget created!")
    
    # Step 2: Add interaction endpoint
    await widget.interaction(
        endpoint="/api/v1/widgets/weather/update",
        headers={"Content-Type": "application/json"},
        refresh_interval=600,  # Refresh every 10 minutes
        func="""
def process_data(widget_data, request, user):
    # Process weather API response
    temp = widget_data.get('main', {}).get('temp', 0)
    description = widget_data.get('weather', [{}])[0].get('description', 'N/A')
    humidity = widget_data.get('main', {}).get('humidity', 0)
    
    return {
        'temperature': f"{temp}°C",
        'conditions': description.title(),
        'humidity': f"{humidity}%",
        'status': 'updated'
    }
        """
    )
    
    print("✅ Interaction added!")
    print(f"   Endpoint: /api/v1/widgets/weather/update")
    print(f"   Refresh: Every 600 seconds")
    
    # Step 3: Save to MongoDB
    await widget.save()
    
    print("✅ Widget saved to database!")
    print(f"\n📋 Summary:")
    print(f"   Widget ID: {widget.current_widget.id}")
    print(f"   Endpoint: /api/v1/widgets/weather/update")
    print(f"   Components: {len(widget.current_widget.interactions)}")
    
    return widget

if __name__ == "__main__":
    asyncio.run(create_widget_with_interaction())
```

**Run it:**

```bash
python create_widget.py
```

**Expected Output:**

```
🎨 Creating widget with interaction...

✅ Widget created!
✅ Interaction added!
   Endpoint: /api/v1/widgets/weather/update
   Refresh: Every 600 seconds
✅ Widget saved to database!

📋 Summary:
   Widget ID: wgt_weather_abc123
   Endpoint: /api/v1/widgets/weather/update
   Components: 1
```

---

## Part 3: Test Your Widget

### Step 6: Interact with Widget via Public API

Create `test_widget.py`:

```python
import requests
import os
from dotenv import load_dotenv

load_dotenv()

def test_widget_interaction(widget_id: str):
    """Test widget interaction via public API"""
    
    print(f"\n🧪 Testing widget interaction...\n")
    
    # Public API endpoint
    public_url = f"{os.getenv('LAZI_API_URL')}/api/v1/public"
    
    # Interaction request
    interaction_data = {
        "widget_id": widget_id,
        "project_id": os.getenv("PROJECT_ID"),
        "endpoint": "/api/v1/widgets/weather/update",
        "params": {
            "city": "San Francisco",
            "units": "metric"
        }
    }
    
    # Make request
    response = requests.post(
        public_url,
        json=interaction_data,
        headers={
            "Authorization": f"Bearer {os.getenv('LAZI_ACCESS_TOKEN')}",
            "Content-Type": "application/json"
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Widget interaction successful!")
        print(f"   Response: {result}")
    else:
        print(f"❌ Interaction failed: {response.status_code}")
        print(f"   Error: {response.text}")

if __name__ == "__main__":
    # Replace with your actual widget ID
    widget_id = "wgt_weather_abc123"
    test_widget_interaction(widget_id)
```

---

## Part 4: Complete Example

### Full Integration

Create `complete_workflow.py`:

```python
import asyncio
import requests
import os
from dotenv import load_dotenv
from lazi_sdk import WriteWidget

load_dotenv()

class CompleteWidgetWorkflow:
    """Complete OAuth2 → Widget Creation → Testing workflow"""
    
    def __init__(self):
        self.base_url = os.getenv("LAZI_API_URL")
        self.access_token = None
        self.widget_id = None
    
    def authenticate(self):
        """Step 1: OAuth2 authentication"""
        print("\n🔐 Step 1: Authenticating...")
        
        response = requests.post(
            f"{self.base_url}/oauth/token",
            data={
                "grant_type": "password",
                "username": os.getenv("LAZI_USERNAME"),
                "password": os.getenv("LAZI_PASSWORD"),
                "scope": "widgets:write widgets:read projects:read"
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        if response.status_code == 200:
            tokens = response.json()
            self.access_token = tokens["access_token"]
            print("✅ Authenticated successfully!")
            return tokens
        else:
            raise Exception(f"Authentication failed: {response.text}")
    
    async def create_widget(self):
        """Step 2: Create widget"""
        print("\n🎨 Step 2: Creating widget...")
        
        widget = WriteWidget(
            token=self.access_token,
            user_id=os.getenv("LAZI_USER_ID", "default_user")
        )
        
        await widget.create_new(
            name="Tutorial Widget",
            description="Created in OAuth2 tutorial",
            size="medium",
            project_id=os.getenv("PROJECT_ID")
        )
        
        await widget.interaction(
            endpoint="/api/v1/widgets/tutorial/test",
            headers={"Content-Type": "application/json"},
            refresh_interval=0,
            func="def process_data(data, req, user): return {'status': 'success', 'data': data}"
        )
        
        await widget.save()
        self.widget_id = widget.current_widget.id
        
        print(f"✅ Widget created: {self.widget_id}")
        return widget
    
    def test_widget(self):
        """Step 3: Test widget"""
        print("\n🧪 Step 3: Testing widget...")
        
        response = requests.post(
            f"{self.base_url}/api/v1/public",
            json={
                "widget_id": self.widget_id,
                "project_id": os.getenv("PROJECT_ID"),
                "endpoint": "/api/v1/widgets/tutorial/test",
                "params": {"test": "data"}
            },
            headers={"Authorization": f"Bearer {self.access_token}"}
        )
        
        if response.status_code == 200:
            print("✅ Widget test successful!")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Test failed: {response.text}")
    
    async def run(self):
        """Execute complete workflow"""
        print("="*60)
        print("🚀 Complete Widget Creation Workflow")
        print("="*60)
        
        # Step 1: Authenticate
        self.authenticate()
        
        # Step 2: Create widget
        await self.create_widget()
        
        # Step 3: Test widget
        self.test_widget()
        
        print("\n" + "="*60)
        print("✅ Workflow completed successfully!")
        print("="*60)
        print(f"\n📋 Summary:")
        print(f"   Access Token: {self.access_token[:20]}...")
        print(f"   Widget ID: {self.widget_id}")
        print(f"\n🎉 Your widget is ready to use!")

if __name__ == "__main__":
    workflow = CompleteWidgetWorkflow()
    asyncio.run(workflow.run())
```

**Run the complete workflow:**

```bash
python complete_workflow.py
```

**Expected Output:**

```
============================================================
🚀 Complete Widget Creation Workflow
============================================================

🔐 Step 1: Authenticating...
✅ Authenticated successfully!

🎨 Step 2: Creating widget...
✅ Widget created: wgt_tutorial_xyz789

🧪 Step 3: Testing widget...
✅ Widget test successful!
   Response: {'status': 'success', 'data': {'test': 'data'}}

============================================================
✅ Workflow completed successfully!
============================================================

📋 Summary:
   Access Token: eyJhbGciOiJIUzI1NiIs...
   Widget ID: wgt_tutorial_xyz789

🎉 Your widget is ready to use!
```

---

## Troubleshooting

### Common Issues

#### 1. Authentication Failed (401)

**Problem:** Invalid credentials or expired token

**Solution:**
```python
# Check credentials in .env file
LAZI_USERNAME=correct_email@example.com
LAZI_PASSWORD=correct_password

# If token expired, refresh it
python refresh_token.py
```

#### 2. Permission Denied (403)

**Problem:** Insufficient OAuth2 scopes

**Solution:**
```python
# Request required scopes during authentication
scopes = "widgets:write widgets:read projects:read files:write"
```

#### 3. Widget Not Found (404)

**Problem:** Invalid project_id or widget_id

**Solution:**
```python
# Verify project_id exists in your Lazi dashboard
# Check widget_id was returned after creation
print(f"Widget ID: {widget.current_widget.id}")
```

#### 4. Code Validation Error

**Problem:** Invalid Python code in interaction logic

**Solution:**
```python
# Ensure function signature is correct
def process_data(widget_data, request, user):
    # Your logic here
    return {...}

# Test code locally before uploading
```

---

## Next Steps

Now that you've completed the OAuth2 and widget creation workflow, try:

1. **Create advanced widgets** - See [SDK Examples](../sdk/examples.md)
2. **Add components** - Learn about [Widget Components](../api/widgets.md)
3. **Upload media** - Use `widget.attach_media()` for images/files
4. **Build iOS app** - Integrate widgets in SwiftUI
5. **Explore security** - Read [Advanced Security](../advanced/security.md)

---

## Resources

- **API Documentation**: [/docs/api/overview.md](../api/overview.md)
- **SDK Reference**: [/docs/sdk/widgets.md](../sdk/widgets.md)
- **OAuth2 Scopes**: [/docs/advanced/scopes.md](../advanced/scopes.md)
- **GitHub Repository**: [https://github.com/Jacey1225/personal-manager-02](https://github.com/Jacey1225/personal-manager-02)

---

**Questions?** Open an issue on GitHub or contact support@lazi.com

**Happy coding! 🚀**

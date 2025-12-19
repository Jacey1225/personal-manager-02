"""
Comprehensive test suite for Widget API OAuth2 flow and public endpoints.
Tests the complete flow: OAuth2 authentication → Widget creation → Public API interaction

Note: Test data is NOT cleaned up after execution for manual verification.
"""

"""
Comprehensive test suite for Widget API OAuth2 flow and public endpoints.
Tests the complete flow: OAuth2 authentication → Widget creation → Public API interaction

Note: Test data is NOT cleaned up after execution for manual verification.
Authentication is handled directly within the widget write process.

IMPORTANT: Run this test from the project root directory:
    cd /Users/jaceysimpson/Vscode/personal-manager-02
    pytest -v api/tests/test_widget_api_flow.py
"""

import pytest
import logging
import sys
from httpx import AsyncClient, ASGITransport
from api.main import build_app
from api.config.fetchMongo import MongoHandler
import uuid
from datetime import datetime
import json

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_widget_api_flow.log')
    ]
)
logger = logging.getLogger(__name__)

# Test data configuration
TEST_USER_CONFIG = {
    "username": f"test_widget_user_{uuid.uuid4().hex[:8]}",
    "email": f"test_{uuid.uuid4().hex[:8]}@example.com",
    "password": "TestPass1!",
    "project_id": str(uuid.uuid4()),
    "project_name": "Test Widget Project"
}

# Store data across tests
test_context = {
    "access_token": "",
    "user_id": "",
    "widget_id": "",
    "project_id": ""
}


@pytest.fixture(scope="module")
async def async_client():
    """Create async test client for the FastAPI app."""
    logger.info("=" * 80)
    logger.info("INITIALIZING TEST SUITE: Widget API OAuth2 Flow")
    logger.info("=" * 80)
    
    app = build_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        logger.info(f"✓ Test client initialized with base URL: {client.base_url}")
        yield client
    
    logger.info("=" * 80)
    logger.info("TEST SUITE COMPLETED - Data preserved in database for review")
    logger.info("=" * 80)


@pytest.fixture(scope="module")
async def setup_test_user(async_client):
    """Create a test user with developer privileges."""
    logger.info("\n" + "=" * 80)
    logger.info("SETUP: Creating Test User with Developer Privileges")
    logger.info("=" * 80)
    
    # Clean up any existing test user first
    logger.info(f"Cleaning up any existing test data")
    user_config = MongoHandler("userAuthDatabase", "userCredentials")
    try:
        # Delete any existing test users (including ones with bad data)
        existing_users = await user_config.get_multi_doc({"username": {"$regex": "^test_widget_user_"}})
        if isinstance(existing_users, list) and len(existing_users) > 0:
            logger.info(f"Found {len(existing_users)} existing test users to delete")
            for existing_user in existing_users:
                if "user_id" in existing_user:
                    logger.info(f"Deleting test user: {existing_user.get('username', 'unknown')}")
                    await user_config.post_delete({"user_id": existing_user["user_id"]})
                    
                    # Also clean up keyring
                    import keyring
                    try:
                        keyring.delete_password("user_auth", existing_user["user_id"])
                        logger.info(f"  Deleted keyring password for {existing_user['user_id']}")
                    except Exception as e:
                        logger.debug(f"  No keyring entry found: {e}")
        else:
            logger.info("No existing test users found to clean up")
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")
    finally:
        await user_config.close_client()
    
    # Create user
    logger.info(f"Creating user: {TEST_USER_CONFIG['username']}")
    logger.info(f"Email: {TEST_USER_CONFIG['email']}")
    
    response = await async_client.get(
        "/auth/signup",
        params={
            "username": TEST_USER_CONFIG["username"],
            "email": TEST_USER_CONFIG["email"],
            "password": TEST_USER_CONFIG["password"]
        }
    )
    
    logger.info(f"Signup response status: {response.status_code}")
    logger.info(f"Signup response body: {response.json()}")
    
    assert response.status_code == 200, f"User creation failed: {response.text}"
    data = response.json()
    assert data["status"] == "success", f"User creation status failed: {data}"
    
    user_id = data["user_id"]
    test_context["user_id"] = user_id
    logger.info(f"✓ User created successfully with ID: {user_id}")
    
    # Grant developer privileges
    logger.info(f"Granting developer privileges to user: {user_id}")
    user_config = MongoHandler("userAuthDatabase", "userCredentials")
    
    try:
        await user_config.post_update(
            {"user_id": user_id},
            {"developer": True}
        )
        logger.info(f"✓ Developer privileges granted")
        
        # Verify developer status
        user_doc = await user_config.get_single_doc({"user_id": user_id})
        logger.info(f"User developer status: {user_doc.get('developer', False)}")
        assert user_doc.get('developer') is True, "Developer status not set correctly"
        
    finally:
        await user_config.close_client()
    
    # Set hashed password for OAuth2
    logger.info("Setting hashed password for OAuth2 authentication")
    from api.routes.auth.public import OAuthUser
    
    oauth_handler = OAuthUser(TEST_USER_CONFIG["username"], password=TEST_USER_CONFIG["password"])
    hashed_password = oauth_handler.hash_pass()
    
    user_config = MongoHandler("userAuthDatabase", "userCredentials")
    try:
        await user_config.post_update(
            {"user_id": user_id},
            {"hashed_password": hashed_password}
        )
        logger.info("✓ Hashed password set successfully")
    finally:
        await user_config.close_client()
    
    # Create test project
    logger.info(f"\nCreating test project: {TEST_USER_CONFIG['project_name']}")
    project_config = MongoHandler("userAuthDatabase", "openProjects")
    
    project_doc = {
        "project_id": TEST_USER_CONFIG["project_id"],
        "project_name": TEST_USER_CONFIG["project_name"],
        "user_id": user_id,
        "widgets": [],
        "created_at": datetime.utcnow().isoformat()
    }
    
    try:
        result = await project_config.post_insert(project_doc)
        test_context["project_id"] = TEST_USER_CONFIG["project_id"]
        logger.info(f"✓ Project created with ID: {TEST_USER_CONFIG['project_id']}")
        logger.info(f"MongoDB insert ID: {result.inserted_id}")
    finally:
        await project_config.close_client()
    
    logger.info("\n" + "=" * 80)
    logger.info("SETUP COMPLETE")
    logger.info("=" * 80)
    
    yield {
        "user_id": user_id,
        "username": TEST_USER_CONFIG["username"],
        "project_id": TEST_USER_CONFIG["project_id"]
    }


class TestOAuth2Flow:
    """Test OAuth2 authentication flow."""
    
    @pytest.mark.asyncio
    async def test_01_oauth_token_acquisition(self, async_client, setup_test_user):
        """Test OAuth2 password grant flow to obtain access token."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 01: OAuth2 Token Acquisition")
        logger.info("=" * 80)
        logger.info("[LINE 200] Starting OAuth2 token acquisition test")
        
        logger.info(f"[LINE 203] Authenticating user: {TEST_USER_CONFIG['username']}")
        logger.info(f"[LINE 204] Using password grant flow")
        
        # OAuth2 password grant flow with write scope
        logger.info("[LINE 207] Sending POST request to /oauth/token endpoint")
        logger.info("[LINE 208] Requesting 'write' scope for widget creation")
        response = await async_client.post(
            "/oauth/token",
            data={
                "username": TEST_USER_CONFIG["username"],
                "password": TEST_USER_CONFIG["password"],
                "grant_type": "password",
                "scope": "widgets:write"
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        logger.info("[LINE 216] Response received from OAuth2 endpoint")
        logger.info(f"[LINE 217] OAuth2 token response status: {response.status_code}")
        logger.info(f"OAuth2 token response body: {response.json()}")
        
        logger.info("[LINE 220] Validating response status code")
        assert response.status_code == 200, f"OAuth2 token acquisition failed: {response.text}"
        
        logger.info("[LINE 223] Parsing response JSON")
        data = response.json()
        logger.info("[LINE 225] Validating access_token presence")
        assert "access_token" in data, "Access token not in response"
        logger.info("[LINE 227] Validating token_type")
        assert data["token_type"] == "bearer", f"Unexpected token type: {data['token_type']}"
        
        access_token = data["access_token"]
        test_context["access_token"] = access_token
        
        logger.info(f"✓ Access token acquired successfully")
        logger.info(f"Token type: {data['token_type']}")
        logger.info(f"Token preview: {access_token[:20]}...{access_token[-20:]}")
        
        # Decode token to verify payload
        logger.info("[LINE 238] Attempting to decode JWT token")
        import jwt
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        secret_key = os.getenv("JWT_SECRET")
        if secret_key:
            try:
                payload = jwt.decode(access_token, secret_key, algorithms=["HS256"])
                logger.info(f"\nToken payload:")
                logger.info(f"  Subject (user_id): {payload.get('sub')}")
                logger.info(f"  Expiration: {datetime.fromtimestamp(payload.get('exp', 0)).isoformat()}")
                logger.info(f"  Issued at: {datetime.fromtimestamp(payload.get('iat', 0)).isoformat()}")
                logger.info(f"  Scopes: {payload.get('scopes', [])}")
                logger.info(f"  Type: {payload.get('type')}")
            except Exception as e:
                logger.warning(f"Could not decode token: {e}")
        
        logger.info("\n" + "=" * 80)
        logger.info("TEST 01: PASSED ✓")
        logger.info("=" * 80)


class TestWidgetCreation:
    """Test widget creation using the WriteWidget SDK."""
    
    @pytest.mark.asyncio
    async def test_02_create_widget_via_sdk(self, setup_test_user):
        """Test creating a widget using the WriteWidget SDK."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 02: Widget Creation via SDK")
        logger.info("=" * 80)
        logger.info("[LINE 313] Starting widget creation via SDK test")
        
        logger.info("[LINE 315] Validating test prerequisites")
        assert test_context["access_token"], "No access token available"
        assert test_context["project_id"], "No project ID available"
        
        from api.services.lazi_sdk.widgets_write import WriteWidget
        
        username = TEST_USER_CONFIG["username"]
        token = test_context["access_token"]
        project_id = test_context["project_id"]
        
        logger.info(f"Initializing WriteWidget SDK")
        logger.info(f"  Username: {username}")
        logger.info(f"  Project ID: {project_id}")
        logger.info(f"  Token: {token[:20]}...{token[-20:]}")
        
        # Initialize widget writer
        logger.info("[LINE 329] Creating WriteWidget instance")
        widget_writer = WriteWidget(
            username=username,
            token=token,
            project_id=project_id
        )
        logger.info("[LINE 334] WriteWidget instance created successfully")
        
        logger.info(f"\n✓ WriteWidget SDK initialized")
        logger.info(f"  Widget ID: {widget_writer.current_widget.widget_id}")
        
        # Create widget
        widget_name = "Test Counter Widget"
        widget_size = "medium"
        
        logger.info(f"\n[LINE 344] Creating widget: {widget_name}")
        logger.info(f"[LINE 345] Size: {widget_size}")
        
        logger.info("[LINE 347] Calling widget_writer.create()")
        result = await widget_writer.create(
            name=widget_name,
            size=widget_size
        )
        logger.info("[LINE 351] Widget create() method returned")
        
        assert result is True, "Widget creation failed"
        logger.info(f"✓ Widget created successfully")
        logger.info(f"  Widget ID: {widget_writer.current_widget.widget_id}")
        logger.info(f"  Widget name: {widget_writer.current_widget.name}")
        logger.info(f"  Widget size: {widget_writer.current_widget.size}")
        
        test_context["widget_id"] = widget_writer.current_widget.widget_id
        
        # Add interaction endpoint
        logger.info(f"\n[LINE 363] Adding interaction endpoint: /increment")
        
        logger.info("[LINE 365] Defining increment_counter function")
        def increment_counter(context):
            """Increment counter logic."""
            current_count = context.get("count", 0)
            return {"count": current_count + 1}
        
        logger.info("[LINE 370] Calling widget_writer.interaction()")
        interaction = await widget_writer.interaction(
            endpoint="/increment",
            headers={"Content-Type": "application/json"},
            refresh_interval=0,
            func=increment_counter
        )
        
        logger.info(f"✓ Interaction endpoint added")
        logger.info(f"  Endpoint: /increment")
        logger.info(f"  Headers: {interaction.headers}")
        logger.info(f"  Refresh interval: {interaction.refresh_interval}s")
        logger.info(f"  Logic length: {len(interaction.logic)} characters")
        
        # Add component for display
        logger.info(f"\n[LINE 384] Adding component: counter display")
        
        logger.info("[LINE 386] Calling widget_writer.component() for text display")
        component = await widget_writer.component(
            endpoint="/increment",
            type="text",
            content=[{"text": "Counter: 0"}],
            props={
                "fontSize": "24px",
                "fontWeight": "bold",
                "color": "#007AFF",
                "alignment": "center"
            }
        )
        
        logger.info(f"✓ Component added")
        logger.info(f"  Component type: {component.type}")
        logger.info(f"  Component content: {component.content}")
        logger.info(f"  Component properties: {component.properties}")
        
        # Add button component
        logger.info(f"\nAdding component: increment button")
        
        button_component = await widget_writer.component(
            endpoint="/increment",
            type="button",
            content=[{"label": "Increment"}],
            props={
                "backgroundColor": "#007AFF",
                "textColor": "#FFFFFF",
                "borderRadius": "8px",
                "padding": "12px 24px"
            }
        )
        
        logger.info(f"✓ Button component added")
        logger.info(f"  Component type: {button_component.type}")
        logger.info(f"  Component content: {button_component.content}")
        
        # Save widget
        logger.info(f"\n[LINE 422] Saving widget to database and project")
        
        logger.info("[LINE 424] Calling widget_writer.save()")
        await widget_writer.save()
        logger.info("[LINE 426] Widget save() completed")
        
        logger.info(f"✓ Widget saved successfully")
        logger.info(f"  Widget ID: {test_context['widget_id']}")
        logger.info(f"  Associated with project: {project_id}")
        
        # Verify widget was saved
        logger.info(f"\nVerifying widget in database")
        widget_config = MongoHandler("userAuthDatabase", "openWidgets")
        
        try:
            widget_doc = await widget_config.get_single_doc({
                "widget_id": test_context["widget_id"]
            })
            
            assert widget_doc is not None, "Widget not found in database"
            logger.info(f"✓ Widget found in database")
            logger.info(f"  Name: {widget_doc.get('name')}")
            logger.info(f"  Size: {widget_doc.get('size')}")
            logger.info(f"  Interactions: {list(widget_doc.get('interactions', {}).keys())}")
            logger.info(f"  Component count: {len(widget_doc.get('interactions', {}).get('/increment', {}).get('components', []))}")
            
        finally:
            await widget_config.close_client()
        
        # Verify widget added to project
        logger.info(f"\nVerifying widget added to project")
        project_config = MongoHandler("userAuthDatabase", "openProjects")
        
        try:
            project_doc = await project_config.get_single_doc({
                "project_id": project_id
            })
            
            assert project_doc is not None, "Project not found in database"
            assert test_context["widget_id"] in project_doc.get("widgets", []), "Widget not in project's widget list"
            
            logger.info(f"✓ Widget added to project successfully")
            logger.info(f"  Project widgets: {project_doc.get('widgets', [])}")
            
        finally:
            await project_config.close_client()
        
        logger.info("\n" + "=" * 80)
        logger.info("TEST 02: PASSED ✓")
        logger.info("=" * 80)


class TestPublicAPIEndpoints:
    """Test public API endpoints for widget interaction."""
    
    @pytest.mark.asyncio
    async def test_03_public_widget_interaction(self, async_client, setup_test_user):
        """Test calling widget interaction through public API endpoint."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 03: Public Widget Interaction")
        logger.info("=" * 80)
        logger.info("[LINE 476] Starting public widget interaction test")
        
        logger.info("[LINE 478] Validating test context data")
        assert test_context["user_id"], "No user ID available"
        assert test_context["project_id"], "No project ID available"
        assert test_context["widget_id"], "No widget ID available"
        
        user_id = test_context["user_id"]
        project_id = test_context["project_id"]
        widget_id = test_context["widget_id"]
        
        logger.info(f"Preparing public API request")
        logger.info(f"  User ID: {user_id}")
        logger.info(f"  Project ID: {project_id}")
        logger.info(f"  Widget ID: {widget_id}")
        logger.info(f"  Endpoint: /increment")
        
        # Test initial increment
        payload = {
            "user_id": user_id,
            "project_id": project_id,
            "widget_id": widget_id,
            "endpoint": "/increment",
            "headers": {"Content-Type": "application/json"},
            "params": {"count": 0}
        }
        
        logger.info(f"\n[LINE 503] Request payload:")
        logger.info(json.dumps(payload, indent=2))
        
        logger.info("[LINE 506] Sending POST request to /public endpoint")
        response = await async_client.post(
            "/public",
            json=payload
        )
        logger.info("[LINE 510] Response received from /public endpoint")
        
        logger.info(f"\nPublic API response status: {response.status_code}")
        logger.info(f"Public API response body: {response.text}")
        
        assert response.status_code == 200, f"Public API call failed: {response.text}"
        
        logger.info(f"✓ Widget interaction executed successfully")
        
        # Test multiple increments
        logger.info(f"\nTesting multiple increments")
        
        for i in range(1, 6):
            payload["params"]["count"] = i
            logger.info(f"  Increment #{i+1} - Count: {i}")
            
            response = await async_client.post(
                "/public",
                json=payload
            )
            
            assert response.status_code == 200, f"Increment #{i+1} failed: {response.text}"
            logger.info(f"  ✓ Response status: {response.status_code}")
        
        logger.info(f"\n✓ All widget interactions completed successfully")
        
        logger.info("\n" + "=" * 80)
        logger.info("TEST 03: PASSED ✓")
        logger.info("=" * 80)
    
    @pytest.mark.asyncio
    async def test_04_public_api_error_handling(self, async_client, setup_test_user):
        """Test public API error handling for invalid requests."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 04: Public API Error Handling")
        logger.info("=" * 80)
        logger.info("[LINE 538] Starting public API error handling test")
        
        # Test with non-existent widget
        logger.info(f"\n[LINE 541] Test 5a: Non-existent widget")
        
        payload = {
            "user_id": test_context["user_id"],
            "project_id": test_context["project_id"],
            "widget_id": "non_existent_widget_id",
            "endpoint": "/increment",
            "headers": {},
            "params": {}
        }
        
        logger.info(f"Request payload: {json.dumps(payload, indent=2)}")
        
        response = await async_client.post("/public", json=payload)
        
        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Response body: {response.json()}")
        
        assert response.status_code == 404, "Expected 404 for non-existent widget"
        logger.info(f"✓ Correctly returned 404 for non-existent widget")
        
        # Test with widget not in project
        logger.info(f"\nTest 5b: Widget not in project")
        
        # Create a widget not associated with the project
        other_widget_id = str(uuid.uuid4())
        widget_config = MongoHandler("userAuthDatabase", "openWidgets")
        
        try:
            await widget_config.post_insert({
                "widget_id": other_widget_id,
                "name": "Orphan Widget",
                "interactions": {
                    "/test": {
                        "logic": "def test_func(context):\n    return {}",
                        "components": []
                    }
                }
            })
            logger.info(f"Created orphan widget: {other_widget_id}")
        finally:
            await widget_config.close_client()
        
        payload["widget_id"] = other_widget_id
        payload["endpoint"] = "/test"
        
        logger.info(f"Request payload: {json.dumps(payload, indent=2)}")
        
        response = await async_client.post("/public", json=payload)
        
        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Response body: {response.json()}")
        
        assert response.status_code == 404, "Expected 404 for widget not in project"
        logger.info(f"✓ Correctly returned 404 for widget not in project")
        
        logger.info("\n" + "=" * 80)
        logger.info("TEST 04: PASSED ✓")
        logger.info("=" * 80)


class TestDataPersistence:
    """Verify test data persistence for manual review."""
    
    @pytest.mark.asyncio
    async def test_05_verify_data_persistence(self):
        """Verify all test data is persisted in database."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 05: Data Persistence Verification")
        logger.info("=" * 80)
        logger.info("[LINE 605] Starting data persistence verification test")
        
        logger.info(f"\n[LINE 607] Verifying test data remains in database for review")
        
        # Check user
        logger.info(f"\n[LINE 610] Checking user data:")
        logger.info("[LINE 611] Initializing MongoHandler for userCredentials")
        user_config = MongoHandler("userAuthDatabase", "userCredentials")
        
        try:
            logger.info("[LINE 615] Querying user document from database")
            user = await user_config.get_single_doc({
                "user_id": test_context["user_id"]
            })
            assert user is not None, "User not found"
            logger.info(f"  ✓ User persisted: {user.get('username')}")
            logger.info(f"    User ID: {user.get('user_id')}")
            logger.info(f"    Email: {user.get('email')}")
            logger.info(f"    Developer: {user.get('developer')}")
        finally:
            await user_config.close_client()
        
        # Check project
        logger.info(f"\nChecking project data:")
        project_config = MongoHandler("userAuthDatabase", "openProjects")
        
        try:
            project = await project_config.get_single_doc({
                "project_id": test_context["project_id"]
            })
            assert project is not None, "Project not found"
            logger.info(f"  ✓ Project persisted: {project.get('project_name')}")
            logger.info(f"    Project ID: {project.get('project_id')}")
            logger.info(f"    Widgets: {project.get('widgets', [])}")
        finally:
            await project_config.close_client()
        
        # Check widget
        logger.info(f"\nChecking widget data:")
        widget_config = MongoHandler("userAuthDatabase", "openWidgets")
        
        try:
            widget = await widget_config.get_single_doc({
                "widget_id": test_context["widget_id"]
            })
            assert widget is not None, "Widget not found"
            logger.info(f"  ✓ Widget persisted: {widget.get('name')}")
            logger.info(f"    Widget ID: {widget.get('widget_id')}")
            logger.info(f"    Size: {widget.get('size')}")
            logger.info(f"    Interactions: {list(widget.get('interactions', {}).keys())}")
        finally:
            await widget_config.close_client()
        
        # Summary
        logger.info(f"\n" + "=" * 80)
        logger.info("DATA PERSISTENCE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"\nAll test data has been preserved for manual review:")
        logger.info(f"\n  User:")
        logger.info(f"    - Username: {TEST_USER_CONFIG['username']}")
        logger.info(f"    - User ID: {test_context['user_id']}")
        logger.info(f"    - Email: {TEST_USER_CONFIG['email']}")
        logger.info(f"\n  Project:")
        logger.info(f"    - Name: {TEST_USER_CONFIG['project_name']}")
        logger.info(f"    - Project ID: {test_context['project_id']}")
        logger.info(f"\n  Widget:")
        logger.info(f"    - Name: Test Counter Widget")
        logger.info(f"    - Widget ID: {test_context['widget_id']}")
        logger.info(f"\n  Database Collections:")
        logger.info(f"    - userAuthDatabase.userCredentials")
        logger.info(f"    - userAuthDatabase.openProjects")
        logger.info(f"    - userAuthDatabase.openWidgets")
        logger.info(f"\nTo clean up test data, manually delete documents with IDs above.")
        
        logger.info("\n" + "=" * 80)
        logger.info("TEST 05: PASSED ✓")
        logger.info("=" * 80)


# Test execution order
pytest_plugins = ['pytest_asyncio']


if __name__ == "__main__":
    """Run tests directly with detailed output."""
    logger.info("Starting Widget API Flow Tests")
    logger.info(f"Test configuration: {TEST_USER_CONFIG}")
    pytest.main([
        __file__,
        "-v",
        "-s",
        "--log-cli-level=INFO",
        "--tb=short"
    ])

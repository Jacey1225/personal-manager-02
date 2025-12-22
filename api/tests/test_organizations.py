import sys
import os
import logging
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from api.main import app
from api.config.fetchMongo import MongoHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

client = TestClient(app)

# Test data tracking
test_users_created = []
test_organizations_created = []

async def cleanup_test_data():
    """Clean up test data after tests"""
    logger.info("Cleaning up test data...")
    
    # Clean up organizations
    if test_organizations_created:
        org_handler = MongoHandler("userAuthDatabase", "openOrganizations")
        try:
            await org_handler.get_client()
            for org_id in test_organizations_created:
                await org_handler.post_delete({"id": org_id})
                logger.info(f"Cleaned up test organization: {org_id}")
            await org_handler.close_client()
        except Exception as e:
            logger.error(f"Error cleaning up organizations: {e}")
        finally:
            test_organizations_created.clear()
    
    # Clean up users
    if test_users_created:
        user_handler = MongoHandler("userAuthDatabase", "userCredentials")
        try:
            await user_handler.get_client()
            for user_id in test_users_created:
                await user_handler.post_delete({"user_id": user_id})
                logger.info(f"Cleaned up test user: {user_id}")
            await user_handler.close_client()
        except Exception as e:
            logger.error(f"Error cleaning up users: {e}")
        finally:
            test_users_created.clear()

@pytest.fixture
async def setup_test_user():
    """Create a test user before organization tests"""
    logger.info("Setting up test user for organization tests")
    
    test_user_id = "test_org_user_123"
    test_username = "org_test_user"
    
    user_handler = MongoHandler("userAuthDatabase", "userCredentials")
    await user_handler.get_client()
    
    # Create test user
    test_user = {
        "username": test_username,
        "email": "orgtest@example.com",
        "user_id": test_user_id,
        "services": {"google": False, "apple": False},
        "projects": [],
        "organizations": []
    }
    await user_handler.post_insert(test_user)
    test_users_created.append(test_user_id)
    await user_handler.close_client()
    
    logger.info(f"Test user created: {test_user_id}")
    
    yield test_user_id
    
    # Cleanup after test
    await cleanup_test_data()

client = TestClient(app)

# Note: The tests for this router are written with the assumption that the endpoints
# with multiple Pydantic models as parameters are refactored to have only one body
# parameter, and the rest of the parameters are passed as query parameters.

@pytest.mark.asyncio
async def test_create_organization_real_db():
    """Test organization creation with real database"""
    logger.info("Starting test_create_organization_real_db")
    
    try:
        # Create test user first
        test_user_id = "test_org_user_123"
        test_username = "org_test_user"
        
        user_handler = MongoHandler("userAuthDatabase", "userCredentials")
        await user_handler.get_client()
        
        test_user = {
            "username": test_username,
            "email": "orgtest@example.com",
            "user_id": test_user_id,
            "services": {"google": False, "apple": False},
            "projects": [],
            "organizations": []
        }
        await user_handler.post_insert(test_user)
        test_users_created.append(test_user_id)
        await user_handler.close_client()
        
        logger.info(f"Test user created: {test_user_id}")
        
        # Create organization
        org_data = {
            "id": "",
            "name": "Test Org",
            "members": [],
            "projects": []
        }
        
        response = client.post(f"/organizations/create_org?user_id={test_user_id}", json=org_data)
        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response JSON: {response.json()}")
        
        assert response.status_code == 200
        org_id = response.json()
        assert isinstance(org_id, str)
        test_organizations_created.append(org_id)
        
        # Verify organization was created in database
        org_handler = MongoHandler("userAuthDatabase", "openOrganizations")
        await org_handler.get_client()
        created_org = await org_handler.get_single_doc({"id": org_id})
        await org_handler.close_client()
        
        assert created_org is not None
        assert created_org["name"] == "Test Org"
        
        logger.info("Finished test_create_organization_real_db")
    finally:
        await cleanup_test_data()

@patch('api.routes.organization_router.commander.create_organization', new_callable=AsyncMock)
def test_create_organization_mocked(mock_create_organization):
    logger.info("Starting test_create_organization_mocked")
    mock_create_organization.return_value = "test_org_id_123"
    
    org_data = {
        "id": "",
        "name": "Test Org",
        "members": ["user1"],
        "projects": []
    }
    
    response = client.post("/organizations/create_org?user_id=test_user", json=org_data)
    logger.info(f"Response status code: {response.status_code}")
    logger.info(f"Response JSON: {response.json()}")
    
    assert response.status_code == 200
    assert response.json() == "test_org_id_123"
    logger.info("Finished test_create_organization_mocked")

@patch('api.routes.organization_router.commander.delete_organization', new_callable=AsyncMock)
def test_delete_organization(mock_delete_organization):
    logger.info("Starting test_delete_organization")
    mock_delete_organization.return_value = True
    
    response = client.delete("/organizations/delete_org?user_id=test_user&organization_id=test_org&force_refresh=false")
    logger.info(f"Response status code: {response.status_code}")
    logger.info(f"Response JSON: {response.json()}")
    
    assert response.status_code == 200
    assert response.json() == True
    logger.info("Finished test_delete_organization")

@patch('api.routes.organization_router.commander.list_organizations', new_callable=AsyncMock)
def test_list_organizations(mock_list_organizations):
    logger.info("Starting test_list_organizations")
    mock_list_organizations.return_value = [
        {
            "id": "org_123",
            "name": "Test Org",
            "members": ["user1", "user2"],
            "projects": ["project1"]
        }
    ]
    
    response = client.get("/organizations/list_orgs?user_id=test_user&organization_id=&force_refresh=false")
    logger.info(f"Response status code: {response.status_code}")
    logger.info(f"Response JSON: {response.json()}")
    
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["name"] == "Test Org"
    logger.info("Finished test_list_organizations")

@patch('api.routes.organization_router.commander.add_member', new_callable=AsyncMock)
def test_add_member(mock_add_member):
    logger.info("Starting test_add_member")
    mock_add_member.return_value = True
    
    response = client.post("/organizations/add_member?user_id=test_user&organization_id=test_org&force_refresh=false&new_email=test@test.com&new_username=testuser")
    logger.info(f"Response status code: {response.status_code}")
    logger.info(f"Response JSON: {response.json()}")
    
    assert response.status_code == 200
    assert response.json() == True
    logger.info("Finished test_add_member")

@patch('api.routes.organization_router.commander.remove_member', new_callable=AsyncMock)
def test_remove_member(mock_remove_member):
    logger.info("Starting test_remove_member")
    mock_remove_member.return_value = True
    
    response = client.delete("/organizations/remove_member?user_id=test_user&organization_id=test_org&force_refresh=false&email=test@test.com")
    logger.info(f"Response status code: {response.status_code}")
    logger.info(f"Response JSON: {response.json()}")
    
    assert response.status_code == 200
    assert response.json() == True
    logger.info("Finished test_remove_member")

@patch('api.routes.organization_router.commander.add_project', new_callable=AsyncMock)
def test_add_project(mock_add_project):
    logger.info("Starting test_add_project")
    mock_add_project.return_value = True
    
    project_data = {
        "project_name": "Test Project",
        "project_id": "proj1",
        "widgets": [],
        "project_likes": 0,
        "project_transparency": True,
        "project_members": ["user1"],
        "organizations": []
    }
    
    response = client.post("/organizations/add_project?user_id=test_user&organization_id=test_org&force_refresh=false", json=project_data)
    logger.info(f"Response status code: {response.status_code}")
    logger.info(f"Response JSON: {response.json()}")
    
    assert response.status_code == 200
    assert response.json() == True
    logger.info("Finished test_add_project")

@patch('api.routes.organization_router.commander.remove_project', new_callable=AsyncMock)
def test_remove_project(mock_remove_project):
    logger.info("Starting test_remove_project")
    mock_remove_project.return_value = True
    
    response = client.delete("/organizations/remove_project?user_id=test_user&organization_id=test_org&force_refresh=false&project_id=proj1")
    logger.info(f"Response status code: {response.status_code}")
    logger.info(f"Response JSON: {response.json()}")
    
    assert response.status_code == 200
    assert response.json() == True
    logger.info("Finished test_remove_project")
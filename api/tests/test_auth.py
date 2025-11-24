import sys
import os
import logging
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import httpx
import pytest_asyncio

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from api.main import app
from api.config.fetchMongo import MongoHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Synchronous client for mocked tests
client = TestClient(app)

# Base URL for async client
BASE_URL = "http://testserver"

# Test data cleanup
test_users_created = []
test_projects_created = []

async def cleanup_test_data():
    """Clean up test data after each test"""
    logger.info("Cleaning up test data...")
    
    # Clean up users
    if test_users_created:
        user_handler = MongoHandler(None, "userAuthDatabase", "userCredentials")
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
    
    # Clean up projects  
    if test_projects_created:
        project_handler = MongoHandler(None, "userProjectsDatabase", "projects")
        try:
            await project_handler.get_client()
            for project_id in test_projects_created:
                await project_handler.post_delete({"project_id": project_id})
                logger.info(f"Cleaned up test project: {project_id}")
            await project_handler.close_client()
        except Exception as e:
            logger.error(f"Error cleaning up projects: {e}")
        finally:
            test_projects_created.clear()

# --- Auth Router Tests with Real Database ---

@patch('api.routes.auth_router.keyring')
@pytest.mark.asyncio
async def test_signup_success_real_db(mock_keyring):
    """
    Test successful user signup with real database interactions.
    """
    logger.info("Starting test_signup_success_real_db")
    
    try:
        # Mock keyring only, let database operations be real
        mock_keyring.set_password = MagicMock()
        mock_keyring.get_password = MagicMock(return_value=None)  # No existing password
        
        test_username = "testuser_real"
        test_email = "testreal@example.com" 
        test_password = "password123"
        
        # Use sync client for API calls, but async operations for database validation
        response = client.get(f"/auth/signup?username={test_username}&email={test_email}&password={test_password}")
        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response JSON: {response.json()}")
        
        assert response.status_code == 200
        json_response = response.json()
        assert json_response["status"] == "success"
        assert "user_id" in json_response
        
        # Track user for cleanup
        user_id = json_response["user_id"]
        test_users_created.append(user_id)
        
        # Verify user was actually created in database
        user_handler = MongoHandler(None, "userAuthDatabase", "userCredentials")
        await user_handler.get_client()
        created_user = await user_handler.get_single_doc({"user_id": user_id})
        await user_handler.close_client()
        
        assert created_user is not None
        assert created_user["username"] == test_username
        assert created_user["email"] == test_email
        
        mock_keyring.set_password.assert_called_once()
        logger.info("Finished test_signup_success_real_db")
    finally:
        # Ensure cleanup happens
        await cleanup_test_data()

@patch('api.routes.auth_router.keyring')  
@pytest.mark.asyncio
async def test_signup_user_exists_real_db(mock_keyring):
    """
    Test signup when user already exists using real database.
    """
    logger.info("Starting test_signup_user_exists_real_db")
    
    try:
        mock_keyring.set_password = MagicMock()
        mock_keyring.get_password = MagicMock(return_value=None)
        
        # First create a user
        test_username = "existing_user"
        test_email = "existing@example.com"
        
        user_handler = MongoHandler(None, "userAuthDatabase", "userCredentials")
        await user_handler.get_client()
        
        # Insert user directly into database
        existing_user = {
            "username": test_username,
            "email": test_email,
            "user_id": "existing_user_id",
            "services": {"google": False, "apple": False},
            "projects": {},
            "organizations": []
        }
        await user_handler.post_insert(existing_user)
        test_users_created.append("existing_user_id")
        await user_handler.close_client()
        
        # Now try to signup with same username
        response = client.get(f"/auth/signup?username={test_username}&email=different@example.com&password=password")
        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response JSON: {response.json()}")
        
        assert response.status_code == 200
        assert response.json() == {"status": "failed", "message": "Email or username already exists"}
        logger.info("Finished test_signup_user_exists_real_db")
    finally:
        # Ensure cleanup happens
        await cleanup_test_data()

@patch('api.routes.auth_router.keyring')
@pytest.mark.asyncio  
async def test_login_success_real_db(mock_keyring):
    """
    Test successful user login with real database interactions.
    """
    logger.info("Starting test_login_success_real_db")
    
    try:
        test_username = "loginuser"
        test_password = "loginpass123"
        test_user_id = "login_test_user_id"
        
        # Mock keyring to return our test password
        mock_keyring.get_password = MagicMock(return_value=test_password)
        
        # Create user in database first
        user_handler = MongoHandler(None, "userAuthDatabase", "userCredentials")
        await user_handler.get_client()
        
        test_user = {
            "username": test_username,
            "email": "loginuser@example.com",
            "user_id": test_user_id,
            "services": {"google": False, "apple": False},
            "projects": {},
            "organizations": []
        }
        await user_handler.post_insert(test_user)
        test_users_created.append(test_user_id)
        await user_handler.close_client()
        
        # Test login
        response = client.get(f"/auth/login?username={test_username}&password={test_password}")
        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response JSON: {response.json()}")
        
        assert response.status_code == 200
        assert response.json() == {"status": "success", "user_id": test_user_id}
        
        mock_keyring.get_password.assert_called_once_with("usr_auth", test_username)
        logger.info("Finished test_login_success_real_db")
    finally:
        # Ensure cleanup happens
        await cleanup_test_data()

@patch('api.routes.auth_router.keyring')
@pytest.mark.asyncio
async def test_remove_user_real_db(mock_keyring):
    """
    Test user removal with real database interactions.
    """
    logger.info("Starting test_remove_user_real_db")
    
    try:
        test_user_id = "remove_test_user_id"
        
        # Create user in database first
        user_handler = MongoHandler(None, "userAuthDatabase", "userCredentials") 
        await user_handler.get_client()
        
        test_user = {
            "username": "removeuser",
            "email": "removeuser@example.com",
            "user_id": test_user_id,
            "services": {"google": False, "apple": False},
            "projects": {},
            "organizations": []
        }
        await user_handler.post_insert(test_user)
        await user_handler.close_client()
        
        # Test removal
        request_data = {"user_id": test_user_id}
        response = client.post("/auth/remove_user", json=request_data)
        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response JSON: {response.json()}")
        
        assert response.status_code == 200
        assert response.json() == {"status": "success", "message": "User removed successfully"}
        
        # Verify user was actually removed from database
        await user_handler.get_client()
        removed_user = await user_handler.get_single_doc({"user_id": test_user_id})
        await user_handler.close_client()
        
        assert removed_user == {} or removed_user is None  # User should not exist
        logger.info("Finished test_remove_user_real_db")
    finally:
        # Ensure cleanup happens (though user should be removed by the test itself)
        await cleanup_test_data()
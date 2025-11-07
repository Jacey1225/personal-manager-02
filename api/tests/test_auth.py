import sys
import os
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from api.main import app

client = TestClient(app)

# --- Auth Router Tests ---

@patch('api.routes.auth_router.user_handler')
@patch('api.routes.auth_router.keyring')
def test_signup_success(mock_keyring, mock_user_handler):
    """
    Test successful user signup.
    """
    mock_user_handler.get_single_doc.return_value = None
    mock_user_handler.post_insert.return_value = MagicMock(inserted_id="some_id")
    
    response = client.get("/auth/signup?username=testuser&email=test@example.com&password=password")
    
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "success"
    assert "user_id" in json_response
    mock_keyring.set_password.assert_called_once()

@patch('api.routes.auth_router.user_handler')
def test_signup_user_exists(mock_user_handler):
    """
    Test signup when user already exists.
    """
    mock_user_handler.get_single_doc.return_value = {"username": "testuser", "email": "test@example.com"}
    
    response = client.get("/auth/signup?username=testuser&email=test@example.com&password=password")
    
    assert response.status_code == 200
    assert response.json() == {"status": "failed", "message": "Email or username already exists"}

@patch('api.routes.auth_router.user_handler')
@patch('api.routes.auth_router.keyring')
def test_login_success(mock_keyring, mock_user_handler):
    """
    Test successful user login.
    """
    mock_keyring.get_password.return_value = "password"
    mock_user_handler.get_single_doc.return_value = {"user_id": "test_user_id", "username": "testuser"}
    
    response = client.get("/auth/login?username=testuser&password=password")
    
    assert response.status_code == 200
    assert response.json() == {"status": "success", "user_id": "test_user_id"}

@patch('api.routes.auth_router.keyring')
def test_login_invalid_password(mock_keyring):
    """
    Test login with an invalid password.
    """
    mock_keyring.get_password.return_value = "wrong_password"
    
    response = client.get("/auth/login?username=testuser&password=password")
    
    assert response.status_code == 200
    assert response.json() == {"status": "failed", "message": "Invalid username or password"}

@patch('api.routes.auth_router.keyring')
def test_login_user_not_found(mock_keyring):
    """
    Test login with a username that does not exist.
    """
    mock_keyring.get_password.return_value = None
    
    response = client.get("/auth/login?username=nonexistentuser&password=password")
    
    assert response.status_code == 200
    assert response.json() == {"status": "failed", "message": "Invalid username or password"}

@patch('api.routes.auth_router.keyring')
def test_set_icloud_user(mock_keyring):
    mock_keyring.get_password.return_value = None
    
    request_data = {
        "service_name": "test_service",
        "apple_user": "test_user",
        "apple_pass": "test_pass"
    }
    response = client.post("/auth/set_icloud_user", json=request_data)
    
    assert response.status_code == 200
    assert response.json() == {"status": "success", "message": "iCloud user set successfully"}

@patch('api.routes.auth_router.user_handler', new_callable=AsyncMock)
@patch('api.routes.auth_router.os')
def test_remove_user(mock_os, mock_user_handler):
    mock_user_handler.post_delete.return_value = None
    mock_os.path.exists.return_value = True
    
    request_data = {"user_id": "test_user"}
    response = client.post("/auth/remove_user", json=request_data)
    
    assert response.status_code == 200
    assert response.json() == {"status": "success", "message": "User removed successfully"}
    mock_os.remove.assert_called_once()

@patch('api.routes.auth_router.ConfigureGoogleAPI')
@patch('api.routes.auth_router.os.path.exists')
def test_google_auth_required(mock_exists, mock_google_api):
    mock_exists.return_value = True
    mock_google_api_instance = mock_google_api.return_value
    mock_google_api_instance.enable.return_value = "http://auth.url"
    
    response = client.get("/auth/google?user_id=test_user")
    
    assert response.status_code == 200
    assert response.json() == {"status": "auth_required", "auth_url": "http://auth.url"}

@patch('api.routes.auth_router.ConfigureGoogleAPI')
def test_complete_google_auth(mock_google_api):
    mock_google_api_instance = mock_google_api.return_value
    mock_google_api_instance.complete_auth_flow.return_value = ("cred1", "cred2")
    
    request_data = {"user_id": "test_user", "authorization_code": "test_code"}
    response = client.post("/auth/google/complete", json=request_data)
    
    assert response.status_code == 200
    assert response.json() == {"status": "success", "message": "Google authentication completed successfully"}
import sys
import os
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

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

import sys
import os
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from api.main import app

client = TestClient(app)

# --- Coordination Router Tests ---

@patch('api.resources.coordination_model.user_config')
def test_fetch_users(mock_user_config):
    """
    Test fetching users.
    """
    # Since the endpoint is async, we need to mock the async methods
    async def async_get_single_doc(*args, **kwargs):
        return {"username": "testuser", "email": "test@example.com"}
    
    async def async_get_all(*args, **kwargs):
        return [{"username": "testuser", "email": "test@example.com"}]

    mock_user_config.get_single_doc = MagicMock(side_effect=async_get_single_doc)
    mock_user_config.get_all = MagicMock(side_effect=async_get_all)
    
    request_body = {
        "members": [
            {"email": "test@example.com"}
        ]
    }
    response = client.post("/coordinate/fetch_users", json=request_body)
    
    assert response.status_code == 200
    assert response.json() == {"users": [{"username": "testuser", "email": "test@example.com"}]}

@patch('api.resources.coordination_model.CoordinateDateTimes')
def test_get_availability(MockCoordinateDateTimes):
    """
    Test getting user availability.
    """
    mock_coordinator = MagicMock()
    mock_coordinator.coordinate.return_value = True
    MockCoordinateDateTimes.return_value = mock_coordinator

    request_body = {
        "users": [{"user_id": "user1", "username": "testuser"}],
        "request_start": "2025-01-01T10:00:00",
        "request_end": "2025-01-01T11:00:00"
    }
    
    response = client.post("/coordinate/get_availability", json=request_body)
    
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "success"
    assert json_response["percent_available"] == 100
    assert json_response["users"] == [["testuser", True]]

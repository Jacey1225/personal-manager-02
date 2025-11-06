import sys
import os
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from api.main import app

client = TestClient(app)

# Note: The tests for this router are written with the assumption that the endpoints
# with multiple Pydantic models as parameters are refactored to have only one body
# parameter, and the rest of the parameters are passed as query parameters.

@patch('api.routes.discussion_router.commander.view_discussion', new_callable=AsyncMock)
def test_view_discussion(mock_view_discussion):
    mock_view_discussion.return_value = {"discussion": {"title": "Test Discussion"}}
    
    response = client.get("/discussions/view_discussion?user_id=test_user&project_id=test_project&discussion_id=test_discussion")
    
    assert response.status_code == 200
    assert response.json() == {"discussion": {"title": "Test Discussion"}}
    mock_view_discussion.assert_called_once()

@patch('api.routes.discussion_router.commander.list_project_discussions', new_callable=AsyncMock)
def test_list_project_discussions(mock_list_discussions):
    mock_list_discussions.return_value = {"discussions": [{"title": "Test Discussion"}]}
    
    response = client.get("/discussions/list_project_discussions?user_id=test_user&project_id=test_project")
    
    assert response.status_code == 200
    assert response.json() == {"discussions": [{"title": "Test Discussion"}]}
    mock_list_discussions.assert_called_once()

@patch('api.routes.discussion_router.commander.create_discussion', new_callable=AsyncMock)
def test_create_discussion(mock_create_discussion):
    mock_create_discussion.return_value = {"status": "Discussion created successfully"}
    
    discussion_data = {
        "title": "Test Discussion",
        "author_id": "test_user",
        "active_contributors": ["test_user"],
        "content": [{"username": "test_user", "message": "Hello", "timestamp": "2025-01-01T12:00:00"}],
        "transparency": True
    }
    
    # This test assumes the endpoint is refactored to take discussion_data as body
    # and other params as query params.
    response = client.post("/discussions/create_discussion?user_id=test_user&project_id=test_project", json=discussion_data)
    
    assert response.status_code == 200
    assert response.json() == {"status": "Discussion created successfully"}

@patch('api.routes.discussion_router.commander.delete_discussion', new_callable=AsyncMock)
def test_delete_discussion(mock_delete_discussion):
    mock_delete_discussion.return_value = {"status": "success"}
    
    response = client.post("/discussions/delete_discussion?user_id=test_user&project_id=test_project&discussion_id=test_discussion")
    
    assert response.status_code == 200
    assert response.json() == {"status": "success"}

@patch('api.routes.discussion_router.commander.add_member_to_discussion', new_callable=AsyncMock)
def test_add_member_to_discussion(mock_add_member):
    mock_add_member.return_value = {"status": "success"}
    
    response = client.post("/discussions/add_member?user_id=test_user&project_id=test_project&discussion_id=test_discussion")
    
    assert response.status_code == 200
    assert response.json() == {"status": "success"}

@patch('api.routes.discussion_router.commander.remove_member_from_discussion', new_callable=AsyncMock)
def test_remove_member_from_discussion(mock_remove_member):
    mock_remove_member.return_value = {"status": "success"}
    
    response = client.post("/discussions/remove_member?user_id=test_user&project_id=test_project&discussion_id=test_discussion")
    
    assert response.status_code == 200
    assert response.json() == {"status": "success"}

@patch('api.routes.discussion_router.commander.post_to_discussion', new_callable=AsyncMock)
def test_post_message(mock_post_to_discussion):
    mock_post_to_discussion.return_value = {"status": "success"}
    
    response = client.post("/discussions/post_message?user_id=test_user&project_id=test_project&discussion_id=test_discussion&message=Hello")
    
    assert response.status_code == 200
    assert response.json() == {"status": "success"}

@patch('api.routes.discussion_router.commander.delete_from_discussion', new_callable=AsyncMock)
def test_remove_message_from_discussion(mock_delete_from_discussion):
    mock_delete_from_discussion.return_value = {"status": "success"}
    
    response = client.post("/discussions/remove_message?user_id=test_user&project_id=test_project&discussion_id=test_discussion&message=Hello")
    
    assert response.status_code == 200
    assert response.json() == {"status": "success"}

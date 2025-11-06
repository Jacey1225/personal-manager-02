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

@patch('api.routes.project_router.commander.view_project', new_callable=AsyncMock)
def test_view_project(mock_view_project):
    mock_view_project.return_value = {"project": {"project_name": "Test Project"}}
    
    response = client.get("/projects/view_project?project_id=test_project&user_id=test_user&project_name=Test%20Project")
    
    assert response.status_code == 200
    assert response.json() == {"project": {"project_name": "Test Project"}}

@patch('api.routes.project_router.commander.like_project', new_callable=AsyncMock)
def test_like_project(mock_like_project):
    mock_like_project.return_value = {"message": "Project liked successfully."}
    
    request_data = {"project_id": "test_project", "user_id": "test_user", "project_name": "Test Project"}
    response = client.post("/projects/like_project", json=request_data)
    
    assert response.status_code == 200
    assert response.json() == {"message": "Project liked successfully."}

@patch('api.routes.project_router.commander.remove_like', new_callable=AsyncMock)
def test_unlike_project(mock_unlike_project):
    mock_unlike_project.return_value = {"message": "Project like removed successfully."}
    
    request_data = {"project_id": "test_project", "user_id": "test_user", "project_name": "Test Project"}
    response = client.post("/projects/unlike_project", json=request_data)
    
    assert response.status_code == 200
    assert response.json() == {"message": "Project like removed successfully."}

@patch('api.routes.project_router.commander.global_delete', new_callable=AsyncMock)
def test_global_delete(mock_global_delete):
    mock_global_delete.return_value = {"message": "All projects deleted successfully."}
    
    request_data = {"project_id": "test_project", "user_id": "test_user", "project_name": "Test Project"}
    response = client.post("/projects/global_delete", json=request_data)
    
    assert response.status_code == 200
    assert response.json() == {"message": "All projects deleted successfully."}

@patch('api.routes.project_router.commander.create_project', new_callable=AsyncMock)
def test_create_project(mock_create_project):
    mock_create_project.return_value = {"message": "Project created successfully."}
    
    request_data = {
        "project_name": "Test Project",
        "project_transparency": True,
        "project_likes": 0,
        "project_members": [["user1", "admin"]],
        "user_id": "test_user"
    }
    response = client.post("/projects/create_project", json=request_data)
    
    assert response.status_code == 200
    assert response.json() == {"message": "Project created successfully."}

@patch('api.routes.project_router.commander.delete_project', new_callable=AsyncMock)
def test_delete_project(mock_delete_project):
    mock_delete_project.return_value = {"message": "Project deleted successfully."}
    
    request_data = {"project_id": "test_project", "user_id": "test_user", "project_name": "Test Project"}
    response = client.post("/projects/delete_project", json=request_data)
    
    assert response.status_code == 200
    assert response.json() == {"message": "Project deleted successfully."}

@patch('api.routes.project_router.commander.rename_project', new_callable=AsyncMock)
def test_rename_project(mock_rename_project):
    mock_rename_project.return_value = {"message": "Project renamed successfully."}
    
    request_data = {"project_id": "test_project", "user_id": "test_user", "project_name": "New Name"}
    response = client.post("/projects/rename_project", json=request_data)
    
    assert response.status_code == 200
    assert response.json() == {"message": "Project renamed successfully."}

@patch('api.routes.project_router.commander.get_project_events', new_callable=AsyncMock)
def test_get_project_events(mock_get_events):
    mock_get_events.return_value = [{"event_name": "Test Event"}]
    
    response = client.get("/projects/events/test_project?user_id=test_user")
    
    assert response.status_code == 200
    assert response.json() == [{"event_name": "Test Event"}]

@patch('api.routes.project_router.commander.add_project_member', new_callable=AsyncMock)
def test_add_project_member(mock_add_member):
    mock_add_member.return_value = {"message": "Member added successfully."}
    
    response = client.get("/projects/add_member?project_id=test_project&user_id=test_user&new_email=test@test.com&new_username=new_user&code=123")
    
    assert response.status_code == 200
    assert response.json() == {"message": "Member added successfully."}

@patch('api.routes.project_router.commander.delete_project_member', new_callable=AsyncMock)
def test_delete_project_member(mock_delete_member):
    mock_delete_member.return_value = {"message": "Member deleted successfully."}
    
    response = client.delete("/projects/delete_member?project_id=test_project&user_id=test_user&email=test@test.com&username=testuser")
    
    assert response.status_code == 200
    assert response.json() == {"message": "Member deleted successfully."}

@patch('api.routes.project_router.commander.list_projects', new_callable=AsyncMock)
def test_list_projects(mock_list_projects):
    mock_list_projects.return_value = [{"project_name": "Test Project"}]
    
    response = client.get("/projects/list?user_id=test_user")
    
    assert response.status_code == 200
    assert response.json() == [{"project_name": "Test Project"}]

@patch('api.routes.project_router.commander.edit_transparency', new_callable=AsyncMock)
def test_edit_project_transparency(mock_edit_transparency):
    mock_edit_transparency.return_value = {"message": "Project transparency updated successfully."}
    
    request_data = {"project_id": "test_project", "user_id": "test_user", "project_name": "Test Project"}
    response = client.post("/projects/edit_transparency?transparency=true", json=request_data)
    
    assert response.status_code == 200
    assert response.json() == {"message": "Project transparency updated successfully."}

@patch('api.routes.project_router.commander.edit_permission', new_callable=AsyncMock)
def test_edit_project_permission(mock_edit_permission):
    mock_edit_permission.return_value = {"message": "User permission updated successfully."}
    
    request_data = {"project_id": "test_project", "user_id": "test_user", "project_name": "Test Project"}
    response = client.post("/projects/edit_permission?email=test@test.com&username=testuser&new_permission=admin", json=request_data)
    
    assert response.status_code == 200
    assert response.json() == {"message": "User permission updated successfully."}
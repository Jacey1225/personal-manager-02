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

@patch('api.routes.organization_router.commander.create_organization', new_callable=AsyncMock)
def test_create_organization(mock_create_organization):
    mock_create_organization.return_value = {"status": "success"}
    
    org_data = {
        "name": "Test Org",
        "members": ["user1"],
        "projects": ["project1"]
    }
    
    response = client.post("/organizations/create_org?user_id=test_user", json=org_data)
    
    assert response.status_code == 200
    assert response.json() == {"status": "success"}

@patch('api.routes.organization_router.commander.delete_organization', new_callable=AsyncMock)
def test_delete_organization(mock_delete_organization):
    mock_delete_organization.return_value = {"status": "success"}
    
    response = client.delete("/organizations/delete_org?user_id=test_user&organization_id=test_org&force_refresh=false")
    
    assert response.status_code == 200
    assert response.json() == {"status": "success"}

@patch('api.routes.organization_router.commander.list_organizations', new_callable=AsyncMock)
def test_list_organizations(mock_list_organizations):
    mock_list_organizations.return_value = [{"name": "Test Org"}]
    
    response = client.get("/organizations/list_orgs?user_id=test_user&organization_id=test_org&force_refresh=false")
    
    assert response.status_code == 200
    assert response.json() == [{"name": "Test Org"}]

@patch('api.routes.organization_router.commander.add_member', new_callable=AsyncMock)
def test_add_member(mock_add_member):
    mock_add_member.return_value = {"status": "success"}
    
    response = client.post("/organizations/add_member?user_id=test_user&organization_id=test_org&force_refresh=false&new_email=test@test.com&new_username=testuser")
    
    assert response.status_code == 200
    assert response.json() == {"status": "success"}

@patch('api.routes.organization_router.commander.remove_member', new_callable=AsyncMock)
def test_remove_member(mock_remove_member):
    mock_remove_member.return_value = {"status": "success"}
    
    response = client.delete("/organizations/remove_member?user_id=test_user&organization_id=test_org&force_refresh=false&email=test@test.com")
    
    assert response.status_code == 200
    assert response.json() == {"status": "success"}

@patch('api.routes.organization_router.commander.add_project', new_callable=AsyncMock)
def test_add_project(mock_add_project):
    mock_add_project.return_value = {"status": "success"}
    
    project_data = {"project_name": "Test Project", "project_id": "proj1"}
    
    response = client.post("/organizations/add_project?user_id=test_user&organization_id=test_org&force_refresh=false", json=project_data)
    
    assert response.status_code == 200
    assert response.json() == {"status": "success"}

@patch('api.routes.organization_router.commander.remove_project', new_callable=AsyncMock)
def test_remove_project(mock_remove_project):
    mock_remove_project.return_value = {"status": "success"}
    
    response = client.delete("/organizations/remove_project?user_id=test_user&organization_id=test_org&force_refresh=false&project_id=proj1")
    
    assert response.status_code == 200
    assert response.json() == {"status": "success"}

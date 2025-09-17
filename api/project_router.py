from fastapi import APIRouter, HTTPException
from src.track_projects.handleProjects import FetchProject, ProjectDetails
from pydantic import BaseModel
from typing import List

project_router = APIRouter()

class CreateProjectRequest(BaseModel):
    project_name: str
    project_members: List[tuple[str, str]]
    user_id: str

@project_router.post("/projects/create_project")
async def create_project(request: CreateProjectRequest):
    """Creates a new project.

    Args:
        request (CreateProjectRequest): The request containing project details and user ID.
    
    Returns:
        dict: A success message with the created project details.
    """
    project_handler = FetchProject(request.user_id)
    project_handler.create_project(request.project_name, request.project_members)
    return {"message": "Project created successfully", "project_name": request.project_name}

@project_router.get("/projects/events/{project_id}")
async def get_project_events(project_id: str, user_id: str):
    """Fetches events associated with a specific project.

    Args:
        project_id (str): The ID of the project to fetch events for.
        user_id (str): The ID of the user requesting the events.

    Returns:
        list[dict]: A list of events associated with the project.
    """
    project_handler = FetchProject(user_id)
    return project_handler.fetch_project_events(project_id)

@project_router.get("/projects/add_member")
async def add_project_member(project_id: str, user_id: str, new_email: str, new_username: str):
    """Adds a new member to an existing project.

    Args:
        project_id (str): The ID of the project to add a member to.
        user_id (str): The ID of the user making the request.
        new_email (str): The email of the new member to add.

    Returns:
        dict: A message indicating the result of the operation.
    """
    project_handler = FetchProject(user_id)
    project_handler.add_project_member(project_id, new_email, new_username)
    return {"message": "Member added successfully."}

@project_router.get("/projects/delete_member")
async def delete_project_member(project_id: str, user_id: str, email: str, username: str):
    """Deletes a member from an existing project.

    Args:
        project_id (str): The ID of the project to delete a member from.
        user_id (str): The ID of the user making the request.
        email (str): The email of the member to delete.
        username (str): The username of the member to delete.

    Returns:
        dict: A message indicating the result of the operation.
    """
    project_handler = FetchProject(user_id)
    project_handler.delete_project_member(project_id, email, username)
    return {"message": "Member deleted successfully."}

@project_router.get("/projects/list")
async def list_projects(user_id: str):
    """Lists all projects for a user.

    Args:
        user_id (str): The ID of the user requesting the project list.

    Returns:
        list[dict]: A list of projects associated with the user.
    """
    project_handler = FetchProject(user_id)
    return project_handler.list_projects()


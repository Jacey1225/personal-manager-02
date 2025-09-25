from src.track_projects.handleProjects import HostActions, GuestActions
from pydantic import BaseModel
from typing import List


class CreateProjectRequest(BaseModel):
    project_name: str
    project_transparency: bool
    project_likes: int
    project_members: List[List[str]]  # Accept arrays from JSON
    user_id: str

class ModifyProjectRequest(BaseModel):
    project_id: str
    user_id: str
    project_name: str


class ProjectModel:
    @staticmethod
    async def create_project(request: CreateProjectRequest):
        """Creates a new project.

        Args:
            request (CreateProjectRequest): The request containing project details and user ID.
        
        Returns:
            dict: A success message with the created project details.
        """
        project_likes = request.project_likes
        transparency = request.project_transparency
        members = request.project_members
        members_list = [(email, username) for email, username in members]
        project_handler = HostActions(request.user_id)
        project_handler.create_project(request.project_name, project_likes, transparency, members_list)
        return {"message": "Project created successfully", "project_name": request.project_name}

    @staticmethod
    async def view_project(request: ModifyProjectRequest):
        """Fetches an existing project.

        Args:
            request (ModifyProjectRequest): The request containing project ID and user ID.

        Returns:
            dict: The details of the requested project.
        """
        user_id = request.user_id
        project_id = request.project_id
        project_handler = GuestActions(user_id)
        project, user_data = project_handler.view_project(project_id)
        return {"project": project, "user_data": user_data}

    @staticmethod
    async def delete_project(request: ModifyProjectRequest):
        """Deletes an existing project.

        Args:
            project_id (str): The ID of the project to delete.
            user_id (str): The ID of the user making the request.

        Returns:
            dict: A message indicating the result of the operation.
        """
        print(f"Deleting project: {request.project_id} for user: {request.user_id}")
        user_id = request.user_id
        project_handler = HostActions(user_id)
        project_handler.delete_project(request.project_id)
        return {"message": "Project deleted successfully."}

    @staticmethod
    async def rename_project(request: ModifyProjectRequest):
        """Renames an existing project.

        Args:
            project_id (str): The ID of the project to rename.
            user_id (str): The ID of the user making the request.
            new_name (str): The new name for the project.

        Returns:
            dict: A message indicating the result of the operation.
        """
        print(f"Renaming project: {request.project_id} to {request.project_name} for user: {request.user_id}")
        user_id = request.user_id
        project_id = request.project_id
        project_name = request.project_name
        project_handler = HostActions(user_id)
        project_handler.rename_project(project_id, project_name)
        return {"message": "Project renamed successfully."}

    @staticmethod
    async def like_project(request: ModifyProjectRequest):
        """Likes an existing project.

        Args:
            request (ModifyProjectRequest): The request containing project ID and user ID.

        Returns:
            dict: A message indicating the result of the operation.
        """
        user_id = request.user_id
        project_id = request.project_id
        project_handler = GuestActions(user_id)
        project_handler.like_project(project_id)
        return {"message": "Project liked successfully."}

    @staticmethod
    async def remove_like(request: ModifyProjectRequest):
        """Removes a like from an existing project.

        Args:
            request (ModifyProjectRequest): The request containing project ID and user ID.

        Returns:
            dict: A message indicating the result of the operation.
        """
        user_id = request.user_id
        project_id = request.project_id
        project_handler = GuestActions(user_id)
        project_handler.remove_like(project_id)
        return {"message": "Project like removed successfully."}

    @staticmethod
    async def get_project_events(request: ModifyProjectRequest):
        """Fetches events associated with a specific project.

        Args:
            project_id (str): The ID of the project to fetch events for.
            user_id (str): The ID of the user requesting the events.

        Returns:
            list[dict]: A list of events associated with the project.
        """
        user_id = request.user_id
        project_id = request.project_id
        project_handler = HostActions(user_id)
        return project_handler.fetch_project_events(project_id)
    
    @staticmethod
    async def add_project_member(request: ModifyProjectRequest, new_email: str, new_username: str):
        """Adds a new member to an existing project.

        Args:
            project_id (str): The ID of the project to add a member to.
            user_id (str): The ID of the user making the request.
            new_email (str): The email of the new member to add.

        Returns:
            dict: A message indicating the result of the operation.
        """
        user_id = request.user_id
        project_id = request.project_id

        project_handler = HostActions(user_id)
        project_handler.add_project_member(project_id, new_email, new_username)
        return {"message": "Member added successfully."}
    
    @staticmethod
    async def delete_project_member(request: ModifyProjectRequest, email: str, username: str):
        """Deletes a member from an existing project.

        Args:
            project_id (str): The ID of the project to delete a member from.
            user_id (str): The ID of the user making the request.
            email (str): The email of the member to delete.
            username (str): The username of the member to delete.

        Returns:
            dict: A message indicating the result of the operation.
        """
        user_id = request.user_id
        project_id = request.project_id
        project_handler = HostActions(user_id)
        project_handler.delete_project_member(project_id, email, username)
        return {"message": "Member deleted successfully."}

    @staticmethod
    async def list_projects(user_id: str):
        """Lists all projects for a user.

        Args:
            user_id (str): The ID of the user requesting the project list.

        Returns:
            list[dict]: A list of projects associated with the user.
        """
        project_handler = HostActions(user_id)
        return project_handler.list_projects()
    
    @staticmethod
    async def edit_transparency(request, new_transparency: bool):
        """Edits the transparency status of an existing project.

        Args:
            project_id (str): The ID of the project to edit.
            new_transparency (bool): The new transparency status to set.

        Returns:
            dict: A message indicating the result of the operation.
        """
        user_id = request.user_id
        project_id = request.project_id
        project_handler = HostActions(user_id)
        project_handler.edit_transparency(project_id, new_transparency)
        return {"message": "Project transparency updated successfully."}
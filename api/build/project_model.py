from api.services.track_projects.handleProjects import HostActions, GuestActions
from api.schemas.projects import CreateProjectRequest, ModifyProjectRequest
from api.config.fetchMongo import MongoHandler
from api.config.cache import project_cache, cached
from typing import List

user_config = MongoHandler(None, "userAuthDatabase", "userCredentials")
project_config = MongoHandler(None, "userAuthDatabase", "openProjects")

class ProjectModel:
    @staticmethod
    async def global_delete(request: ModifyProjectRequest):
        """Deletes all projects for a user.

        Args:
            request (ModifyProjectRequest): The request containing user ID.

        Returns:
            dict: A message indicating the result of the operation.
        """
        user_id = request.user_id
        await user_config.get_client()
        await project_config.get_client()
        handler = await HostActions.fetch(user_id, user_config, project_config)
        await handler.global_delete()
        return {"message": "All projects deleted successfully."}

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
        user_id = request.user_id
        await user_config.get_client()
        await project_config.get_client()
        handler = await HostActions.fetch(user_id, user_config, project_config)
        await handler.create_project(request.project_name, project_likes, transparency, members)
        return {"message": "Project created successfully", "project_name": request.project_name}

    @staticmethod
    @cached(cache=project_cache)
    async def view_project(request: ModifyProjectRequest):
        """Fetches an existing project.

        Args:
            request (ModifyProjectRequest): The request containing project ID and user ID.

        Returns:
            dict: The details of the requested project.
        """
        user_id = request.user_id
        project_id = request.project_id
        await user_config.get_client()
        await project_config.get_client()
        handler = await GuestActions.fetch(user_id, user_config, project_config)
        project, user_data = await handler.view_project(project_id)
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
        await user_config.get_client()
        await project_config.get_client()
        handler = await HostActions.fetch(user_id, user_config, project_config)
        await handler.delete_project(request.project_id)
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
        await user_config.get_client()
        await project_config.get_client()
        handler = await HostActions.fetch(user_id, user_config, project_config)
        await handler.rename_project(project_id, project_name)
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
        await user_config.get_client()
        await project_config.get_client()
        handler = await GuestActions.fetch(user_id, user_config, project_config)
        await handler.like_project(project_id)
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
        await user_config.get_client()
        await project_config.get_client()
        handler = await GuestActions.fetch(user_id, user_config, project_config)
        await handler.remove_like(project_id)
        return {"message": "Project like removed successfully."}

    @staticmethod
    @cached(cache=project_cache)
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
        await user_config.get_client()
        await project_config.get_client()
        handler = await HostActions.fetch(user_id, user_config, project_config)
        return await handler.fetch_project_events(project_id)

    @staticmethod
    async def add_project_member(request: ModifyProjectRequest, new_email: str, new_username: str, code: str):
        """Adds a new member to an existing project.

        Args:
            project_id (str): The ID of the project to add a member to.
            user_id (str): The ID of the user making the request.
            new_email (str): The email of the new member to add.
            new_username (str): The username of the new member to add.
            code (str): The code associated with the project.

        Returns:
            dict: A message indicating the result of the operation.
        """
        user_id = request.user_id
        project_id = request.project_id

        await user_config.get_client()
        await project_config.get_client()
        handler = await GuestActions.fetch(user_id, user_config, project_config)
        await handler.add_project_member(project_id, new_email, new_username, code)
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
        
        # We need to find the user_id for the member being deleted
        # This requires a method to fetch user_id from email and username
        await user_config.get_client()
        await project_config.get_client()
        handler = await GuestActions.fetch(user_id, user_config, project_config)
        target_user_id = await handler.fetch_user_id(email, username)

        if target_user_id:
            await handler.delete_project_member(project_id, target_user_id)
            return {"message": "Member deleted successfully."}
        else:
            return {"message": "Member not found.", "error": True}

    @staticmethod
    async def list_projects(user_id: str):
        """Lists all projects for a user.

        Args:
            user_id (str): The ID of the user requesting the project list.

        Returns:
            list[dict]: A list of projects associated with the user.
        """
        await user_config.get_client()
        await project_config.get_client()
        handler = await HostActions.fetch(user_id, user_config, project_config)
        return await handler.list_projects()

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
        await user_config.get_client()
        await project_config.get_client()
        handler = await HostActions.fetch(user_id, user_config, project_config)
        await handler.edit_transparency(project_id, new_transparency)
        return {"message": "Project transparency updated successfully."}
    
    @staticmethod
    async def edit_permission(request: ModifyProjectRequest, email: str, username: str, new_permission: str):
        """Edits the permission level of a user in an existing project.

        Args:
            project_id (str): The ID of the project to edit.
            email (str): The email of the user to change permissions for.
            username (str): The username of the user to change permissions for.
            new_permission (str): The new permission level to set.

        Returns:
            dict: A message indicating the result of the operation.
        """
        user_id = request.user_id
        project_id = request.project_id
        await user_config.get_client()
        await project_config.get_client()
        handler = await HostActions.fetch(user_id, user_config, project_config)
        await handler.edit_permissions(project_id, email, username, new_permission)
        return {"message": "User permission updated successfully."}
from api.services.track_projects.handleDiscussions import HandleDiscussions, DiscussionData
from api.config.fetchMongo import MongoHandler
from api.schemas.projects import DiscussionRequest
from pydantic import BaseModel, Field
from typing import Optional

user_config = MongoHandler(None, "userAuthDatabase", "userCredentials")
discussion_config = MongoHandler(None, "userAuthDatabase", "openDiscussions")

class DiscussionsModel:
    @staticmethod
    async def view_discussion(request: DiscussionRequest, discussion_id: str) -> dict:
        """Fetches an existing discussion.

        Args:
            request (DiscussionRequest): The request object containing user and project information.
            discussion_id (str): The ID of the discussion to fetch.

        Returns:
            dict: The details of the requested discussion.
        """
        user_id = request.user_id
        project_id = request.project_id
        await user_config.get_client()
        await discussion_config.get_client()
        discussion_control = await HandleDiscussions.fetch(user_id, project_id, user_config, discussion_config)
        discussion = await discussion_control.view_discussion(discussion_id)
        return {"discussion": discussion}

    @staticmethod
    async def list_project_discussions(request: DiscussionRequest) -> dict:
        """Lists all discussions for a project.

        Args:
            request (DiscussionRequest): The request object containing user and project information.

        Returns:
            dict: A list of discussions for the specified project.
        """
        user_id = request.user_id
        project_id = request.project_id
        await user_config.get_client()
        await discussion_config.get_client()
        discussion_control = await HandleDiscussions.fetch(user_id, project_id, user_config, discussion_config)
        discussions = await discussion_control.project_discussions()
        return {"discussions": discussions}

    @staticmethod
    async def create_discussion(request: DiscussionRequest, discussion_data: DiscussionData) -> dict:
        """Creates a new discussion for a project.

        Args:
            request (DiscussionRequest): The request object containing user and project information.

        Returns:
            dict: The created discussion object.
        """
        user_id = request.user_id
        project_id = request.project_id
        await user_config.get_client()
        await discussion_config.get_client()
        discussion_control = await HandleDiscussions.fetch(user_id, project_id, user_config, discussion_config)
        await discussion_control.create_discussion(
            discussion_data.title,
            discussion_data.active_contributors,
            discussion_data.content,
            discussion_data.transparency)
        return {"status": "Discussion created successfully"}

    @staticmethod
    async def delete_discussion(request: DiscussionRequest, discussion_id: str) -> dict:
        """Deletes an existing discussion.

        Args:
            request (DiscussionRequest): The request object containing user and project information.
            discussion_id (str): The ID of the discussion to delete.

        Returns:
            dict: The result of the delete operation.
        """
        user_id = request.user_id
        project_id = request.project_id
        await user_config.get_client()
        await discussion_config.get_client()
        discussion_control = await HandleDiscussions.fetch(user_id, project_id, user_config, discussion_config)
        result = await discussion_control.delete_discussion(discussion_id)
        return result

    @staticmethod
    async def add_member_to_discussion(request: DiscussionRequest, discussion_id: str) -> dict:
        """Adds a new member to an existing discussion.

        Args:
            request (DiscussionRequest): The request object containing user and project information.
            discussion_id (str): The ID of the discussion to add a member to.
            new_username (str): The username of the new member to add.

        Returns:
            dict: The result of the add member operation.
        """
        user_id = request.user_id
        project_id = request.project_id
        await user_config.get_client()
        await discussion_config.get_client()
        discussion_control = await HandleDiscussions.fetch(user_id, project_id, user_config, discussion_config)
        result = await discussion_control.add_member_to_discussion(discussion_id)
        return result

    @staticmethod
    async def remove_member_from_discussion(request: DiscussionRequest, discussion_id: str) -> dict | None:
        """Removes a member from an existing discussion.

        Args:
            request (DiscussionRequest): The request object containing user and project information.
            discussion_id (str): The ID of the discussion to remove a member from.

        Returns:
            dict: The result of the remove member operation.
        """
        user_id = request.user_id
        project_id = request.project_id
        await user_config.get_client()
        await discussion_config.get_client()
        discussion_control = await HandleDiscussions.fetch(user_id, project_id, user_config, discussion_config)
        result = await discussion_control.remove_member_from_discussion(discussion_id)
        return result

    @staticmethod
    async def post_to_discussion(request: DiscussionRequest, discussion_id: str, message: str) -> dict:
        """Posts a message to an existing discussion.

        Args:
            request (DiscussionRequest): The request object containing user and project information.
            discussion_id (str): The ID of the discussion to post to.
            message (str): The content of the message.

        Returns:
            dict: The result of the post operation.
        """
        user_id = request.user_id
        project_id = request.project_id
        await user_config.get_client()
        await discussion_config.get_client()
        discussion_control = await HandleDiscussions.fetch(user_id, project_id, user_config, discussion_config)
        result = await discussion_control.post_to_discussion(discussion_id, user_id, message)
        return result
            
    @staticmethod
    async def delete_from_discussion(request: DiscussionRequest, discussion_id: str, message: str) -> dict | None:
        """Deletes a message from an existing discussion.

        Args:
            request (DiscussionRequest): The request object containing user and project information.
            discussion_id (str): The ID of the discussion to delete from.
            message (str): The content of the message to delete.

        Returns:
            dict: The result of the delete operation.
        """
        user_id = request.user_id
        project_id = request.project_id
        await user_config.get_client()
        await discussion_config.get_client()
        discussion_control = await HandleDiscussions.fetch(user_id, project_id, user_config, discussion_config)
        result = await discussion_control.delete_from_discussion(discussion_id, user_id, message)
        return result
        
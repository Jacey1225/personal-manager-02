from api.config.fetchMongo import MongoHandler
from typing import Optional
import uuid
from datetime import datetime
from api.validation.handleDiscussions import ValidateDiscussions
from api.schemas.projects import PageSchema

validator = ValidateDiscussions()
class HandleDiscussions:
    def __init__(self, 
                 user_id: str, 
                 project_id: str, 
                 user_handler: MongoHandler,
                 widget_handler: MongoHandler):
        self.user_id = user_id
        self.project_id = project_id
        self.user_handler = user_handler
        self.widget_handler = widget_handler
        self.user_data = None

    @classmethod
    async def fetch(cls,
                    user_id: str,
                    project_id: str,
                    user_handler: MongoHandler,
                    widget_handler: MongoHandler):
        self = cls(user_id, project_id, user_handler, widget_handler)
        self.user_data = await self.user_handler.get_single_doc({"user_id": self.user_id})
        return self
    
    async def _migrate_content_format(self, discussion: dict) -> dict:
        """Migrates old tuple-based content format to new dictionary format.
        
        Args:
            discussion (dict): The discussion document
            
        Returns:
            dict: The discussion with migrated content format
        """
        if discussion and "data" in discussion and "content" in discussion["data"]:
            content = discussion["data"]["content"]
            if content and len(content) > 0:
                if isinstance(content[0], list) and len(content[0]) == 2:
                    migrated_content = []
                    for item in content:
                        if isinstance(item, list) and len(item) >= 2:
                            migrated_content.append({
                                "username": item[0],
                                "message": item[1],
                                "timestamp": discussion["data"]["created_time"]  
                            })
                    discussion["data"]["content"] = migrated_content
                    await self.widget_handler.post_update(
                        {"id": discussion["discussion_id"]}, 
                        {"data.content": migrated_content}
                    )
        return discussion

    async def project_discussions(self) -> Optional[list[dict]] | dict:
        """Lists all discussions for the current project.

        Returns:
            Optional[list[dict]] | dict: A list of discussions for the specified project, or an error message.
        """
        discussions = await self.widget_handler.get_multi_doc({"project_id": self.project_id})
        print(f"Fetched discussions for project {self.project_id}: {discussions}")
        
        if discussions:
            discussions = [await self._migrate_content_format(disc) for disc in discussions]
        
        return discussions

    async def create_discussion(self, 
                                title: str, 
                                contributors: list[str], 
                                content: dict) -> dict:
        """Creates a new discussion.

        Args:
            title (str): The title of the discussion.
            contributors (list[str]): A list of user IDs who are contributing to the discussion.
            content (dict): A dictionary containing the message objects with username, message, and timestamp.
            project_id (Optional[str]): The ID of the project that this discussion is related to.

        Returns:
            str: The ID of the created discussion.
        """
        discussion = PageSchema(
            id=str(uuid.uuid4()),
            project_id=self.project_id,
            name=title,
            author_id=self.user_id,
            contributors=contributors,
            creation_time=datetime.now().isoformat(),
            page_type="chat",
            interface_type="chat",
            api_config={},
            metadata=content
        )
        try:
            await self.widget_handler.post_insert(discussion.model_dump())
        except Exception as e:
            return {"status": "error", "message": str(e)}
        
        return {"status": "success", "data": {"page_id": discussion.id}}

    @validator.validate_discussion
    async def delete_discussion(self, discussion_id: str) -> dict:
        """Deletes an existing discussion.

        Args:
            discussion_id (str): The ID of the discussion to delete.
        """
        discussion = await self.widget_handler.get_single_doc({"id": discussion_id})
        if discussion and discussion["data"]["author_id"] == self.user_id:
            query_item = {"id": discussion_id, "data.author_id": self.user_id}
            await self.widget_handler.post_delete(query_item)
            return {"status": "success", "data": {"page_id": discussion_id}}
        else:
            return {"status": "error", "message": "Discussion not found or you don't have permission to delete it"}
    
    @validator.validate_discussion
    async def add_member_to_discussion(self, discussion_id: str):
        """Adds a new member to the discussion.

        Args:
            discussion_id (str): The ID of the discussion to add a member to.
            new_username (str): The username of the new member to add.

        Returns:
            dict: A dictionary containing the status and any relevant data.
        """
        discussion = await self.widget_handler.get_single_doc({"id": discussion_id})
        if discussion and discussion["data"]['author_id'] == self.user_id:
            if self.user_id not in discussion["data"]["active_contributors"]:
                discussion["data"]["active_contributors"].append(self.user_id)
                await self.widget_handler.post_update({"id": discussion_id}, 
                                               {"data.active_contributors": discussion["data"]["active_contributors"]})
                return {"status": "success", "data": {"page_id": discussion_id, "new_member": self.user_id}}
            else:
                return {"status": "error", "message": "User is already a member of the discussion"}
        else:
            return {"status": "error", "message": "Discussion not found"}

    @validator.validate_discussion
    async def remove_member_from_discussion(self, discussion_id: str) -> dict | None:
        """Removes a member from the discussion.

        Args:
            discussion_id (str): The ID of the discussion to remove a member from.

        Returns:
            dict | None: The status and data of the operation.
        """
        discussion = await self.widget_handler.get_single_doc({"id": discussion_id})
        if discussion and discussion["data"]['author_id'] == self.user_id:
            if self.user_id in discussion["data"]["active_contributors"]:
                discussion["data"]["active_contributors"].remove(self.user_id)
                await self.widget_handler.post_update({"id": discussion_id}, 
                                               {"data.active_contributors": discussion["data"]["active_contributors"]})
                return {"status": "success", "data": {"page_id": discussion_id, "removed_member": self.user_id}}
            else:
                return {"status": "error", "message": "User is not a member of the discussion"}

    @validator.validate_discussion
    async def post_to_discussion(self, discussion_id: str, user_id: str, message: str) -> dict:
        """Posts a message to an existing discussion.

        Args:
            discussion_id (str): The ID of the discussion to post to.
            user_id (str): The ID of the user posting the message.
            message (str): The content of the message.
        """
        discussion = await self.widget_handler.get_single_doc({"id": discussion_id})
        if not discussion:
            return {"error": "Discussion not found"}
            
        try:
            username = (await self.user_handler.get_single_doc({"user_id": user_id})).get("username", "Unknown User")

            if username not in discussion["data"]["active_contributors"]:
                discussion["data"]["active_contributors"].append(username)
                await self.widget_handler.post_update(
                    {"id": discussion_id}, 
                    {"data.active_contributors": discussion["data"]["active_contributors"]}
                )
            
            new_message = {
                "username": username,
                "message": message,
                "timestamp": datetime.now().isoformat()
            }
            
            discussion["data"]["content"].append(new_message)
            await self.widget_handler.post_update(
                {"id": discussion_id}, 
                {"data.content": discussion["data"]["content"]}
            )

            return {"status": "success", "data": {"page_id": discussion_id, "message": new_message}}
        except Exception as e:
            return {"error": str(e)}

    @validator.validate_discussion
    async def delete_from_discussion(self, discussion_id: str, user_id: str, message: str) -> dict | None:
        """Deletes a message from a discussion.

        Args:
            discussion_id (str): The ID of the discussion to delete from.
            user_id (str): The ID of the user requesting the deletion.
            message (str): The content of the message to delete.

        Returns:
            dict | None: The status and data of the operation.
        """

        discussion = await self.widget_handler.get_single_doc({"id": discussion_id})
        if discussion:
            try:
                username = (await self.user_handler.get_single_doc({"user_id": user_id})).get("username", "Unknown User")
                content_list = discussion["data"]["content"]
                for i, msg_obj in enumerate(content_list):
                    if (isinstance(msg_obj, dict) and 
                        msg_obj.get("username") == username and 
                        msg_obj.get("message") == message):
                        content_list.pop(i)
                        await self.widget_handler.post_update({"id": discussion_id}, {"data.content": content_list})
                        return {"status": "success", "data": {"page_id": discussion_id}}
                return {"error": "Message not found in discussion"}
            except Exception as e:
                return {"error": str(e)}
        else:
            return {"error": "Discussion not found"}
        
    
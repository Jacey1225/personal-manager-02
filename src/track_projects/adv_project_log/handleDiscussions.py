from src.fetchMongo import MongoHandler
from typing import Optional
from pydantic import BaseModel, Field
import uuid
from datetime import datetime
from src.validators.validators import ValidateDiscussions

user_handler = MongoHandler("userCredentials")
discussion_handler = MongoHandler("openDiscussions")
validator = ValidateDiscussions()

class DiscussionData(BaseModel):
    title: str = Field(description="The title of the discussion")
    author_id: str = Field(description="The User ID of the discussion author")
    active_contributors: list[str] = Field(description="The usernames of everyone who currently contributes to this discussion")
    content: list[dict] = Field(description="A list of message objects with username, message, and timestamp")
    created_time: str = Field(default_factory=lambda: datetime.now().isoformat(), description="The timestamp when the discussion was created")
    transparency: bool = Field(description="The visibility status of the discussion (e.g., True, False)")

class Discussion(BaseModel):
    discussion_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="The UUID of the discussion")
    project_id: str = Field(default="", description="The ID of the project this discussion is linked to, if any")
    data: DiscussionData

class HandleDiscussions:
    def __init__(self, user_id: str, project_id: str):
        self.user_id = user_id
        self.user_data = user_handler.get_single_doc({"user_id": self.user_id})
        self.project_id = project_id

    @validator.validate_discussion
    def view_discussion(self, discussion_id: str) -> dict:
        """Fetches a specific discussion.

        Args:
            discussion_id (str): The ID of the discussion to fetch.

        Returns:
            dict: The details of the requested discussion.
        """
        discussion = discussion_handler.get_single_doc({"discussion_id": discussion_id})
        if discussion:
            # Convert legacy tuple format to new dictionary format if needed
            discussion = self._migrate_content_format(discussion)
        return discussion
    
    def _migrate_content_format(self, discussion: dict) -> dict:
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
                    discussion_handler.post_update(
                        {"discussion_id": discussion["discussion_id"]}, 
                        {"data.content": migrated_content}
                    )
        return discussion

    def project_discussion(self) -> Optional[list[dict]] | dict:
        """Lists all discussions for the current project.

        Returns:
            Optional[list[dict]] | dict: A list of discussions for the specified project, or an error message.
        """
        discussions = discussion_handler.get_multi_doc({"project_id": self.project_id})
        print(f"Fetched discussions for project {self.project_id}: {discussions}")
        
        if discussions:
            discussions = [self._migrate_content_format(disc) for disc in discussions]
        
        return discussions

    def create_discussion(self, title: str, contributors: list[str], content: list[dict], transparency: bool) -> dict:
        """Creates a new discussion.

        Args:
            title (str): The title of the discussion.
            contributors (list[str]): A list of user IDs who are contributing to the discussion.
            content (list[dict]): A list of message objects with username, message, and timestamp.
            transparency (str): The visibility status of the discussion (e.g., public, private).
            project_id (Optional[str]): The ID of the project that this discussion is related to.

        Returns:
            str: The ID of the created discussion.
        """
        if not contributors:
            contributors = [self.user_id]
        if not content:
            content = []
        discussion = Discussion(
            discussion_id=str(uuid.uuid4()),
            project_id=self.project_id,
            data=DiscussionData(
                title=title,
                author_id=self.user_id,
                active_contributors=contributors,
                content=content,
                created_time=datetime.now().isoformat(),
                transparency=transparency,
            )
        )
        try:
            discussion_handler.post_insert(discussion.model_dump())
        except Exception as e:
            return {"status": "error", "message": str(e)}
        
        return {"status": "success", "data": {"discussion_id": discussion.discussion_id}}

    @validator.validate_discussion
    def delete_discussion(self, discussion_id: str) -> dict:
        """Deletes an existing discussion.

        Args:
            discussion_id (str): The ID of the discussion to delete.
        """
        discussion = discussion_handler.get_single_doc({"discussion_id": discussion_id})
        if discussion and discussion["data"]["author_id"] == self.user_id:
            query_item = {"discussion_id": discussion_id, "data.author_id": self.user_id}
            discussion_handler.post_delete(query_item)
            return {"status": "success", "data": {"discussion_id": discussion_id}}
        else:
            return {"status": "error", "message": "Discussion not found or you don't have permission to delete it"}
    
    @validator.validate_discussion
    def add_member_to_discussion(self, discussion_id: str):
        """Adds a new member to the discussion.

        Args:
            discussion_id (str): The ID of the discussion to add a member to.
            new_username (str): The username of the new member to add.

        Returns:
            dict: A dictionary containing the status and any relevant data.
        """
        discussion = discussion_handler.get_single_doc({"discussion_id": discussion_id})
        if discussion and discussion["data"]['author_id'] == self.user_id:
            if self.user_id not in discussion["data"]["active_contributors"]:
                discussion["data"]["active_contributors"].append(self.user_id)
                discussion_handler.post_update({"discussion_id": discussion_id}, {"data.active_contributors": discussion["data"]["active_contributors"]})
                return {"status": "success", "data": {"discussion_id": discussion_id, "new_member": self.user_id}}
            else:
                return {"status": "error", "message": "User is already a member of the discussion"}
        else:
            return {"status": "error", "message": "Discussion not found"}

    @validator.validate_discussion
    def remove_member_from_discussion(self, discussion_id: str) -> dict | None:
        discussion = discussion_handler.get_single_doc({"discussion_id": discussion_id})
        if discussion and discussion["data"]['author_id'] == self.user_id:
            if self.user_id in discussion["data"]["active_contributors"]:
                discussion["data"]["active_contributors"].remove(self.user_id)
                discussion_handler.post_update({"discussion_id": discussion_id}, {"data.active_contributors": discussion["data"]["active_contributors"]})
                return {"status": "success", "data": {"discussion_id": discussion_id, "removed_member": self.user_id}}
            else:
                return {"status": "error", "message": "User is not a member of the discussion"}

    @validator.validate_discussion
    def post_to_discussion(self, discussion_id: str, user_id: str, message: str) -> dict:
        """Posts a message to an existing discussion.

        Args:
            discussion_id (str): The ID of the discussion to post to.
            user_id (str): The ID of the user posting the message.
            message (str): The content of the message.
        """
        discussion = discussion_handler.get_single_doc({"discussion_id": discussion_id})
        if not discussion:
            return {"error": "Discussion not found"}
            
        try:
            username = user_handler.get_single_doc({"user_id": user_id}).get("username", "Unknown User")
            
            if username not in discussion["data"]["active_contributors"]:
                discussion["data"]["active_contributors"].append(username)
                discussion_handler.post_update(
                    {"discussion_id": discussion_id}, 
                    {"data.active_contributors": discussion["data"]["active_contributors"]}
                )
            
            new_message = {
                "username": username,
                "message": message,
                "timestamp": datetime.now().isoformat()
            }
            
            discussion["data"]["content"].append(new_message)
            discussion_handler.post_update(
                {"discussion_id": discussion_id}, 
                {"data.content": discussion["data"]["content"]}
            )
            
            return {"status": "success", "data": {"discussion_id": discussion_id, "message": new_message}}
        except Exception as e:
            return {"error": str(e)}

    @validator.validate_discussion
    def delete_from_discussion(self, discussion_id: str, user_id: str, message: str) -> dict | None:
        discussion = discussion_handler.get_single_doc({"discussion_id": discussion_id})
        if discussion:
            try:
                username = user_handler.get_single_doc({"user_id": user_id}).get("username", "Unknown User")
                content_list = discussion["data"]["content"]
                for i, msg_obj in enumerate(content_list):
                    if (isinstance(msg_obj, dict) and 
                        msg_obj.get("username") == username and 
                        msg_obj.get("message") == message):
                        content_list.pop(i)
                        discussion_handler.post_update({"discussion_id": discussion_id}, {"data.content": content_list})
                        return {"status": "success", "data": {"discussion_id": discussion_id}}
                return {"error": "Message not found in discussion"}
            except Exception as e:
                return {"error": str(e)}
        else:
            return {"error": "Discussion not found"}
        
    
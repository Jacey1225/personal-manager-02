from src.fetchMongo import MongoHandler
from typing import Optional
from pydantic import BaseModel, Field
import uuid
from datetime import datetime

user_handler = MongoHandler("userCredentials")
discussion_handler = MongoHandler("discussions")

class DiscussionData(BaseModel):
    title: str = Field(description="The title of the discussion")
    author_id: str = Field(description="The User ID of the discussion author")
    active_contributors: list[str] = Field(description="The usernames of everyone who currently contributes to this discussion")
    content: list[tuple[str, str]] = Field(description="A list of tuples containing (username, message) for each message in the discussion")
    created_time: str = Field(default_factory=lambda: datetime.now().isoformat(), description="The timestamp when the discussion was created")
    transparency: str = Field(description="The visibility status of the discussion (e.g., public, private)")
    project_id: Optional[str] = Field(description="The ID of the project that this discussion is related to")

class Discussion(BaseModel):
    discussion_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="The UUID of the discussion")
    data: DiscussionData

    def convert_to_dict(self) -> dict:
        return {
            self.discussion_id: self.data.model_dump()
        }

class HandleDiscussions:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.user_data = user_handler.get_single_doc({"user_id": self.user_id})
        self.discussion_data = discussion_handler.get_all() or {}

    def create_discussion(self, title: str, contributors: list[str], content: list[tuple[str, str]], transparency: str, project_id: Optional[str]) -> dict:
        """Creates a new discussion.

        Args:
            title (str): The title of the discussion.
            contributors (list[str]): A list of user IDs who are contributing to the discussion.
            content (list[tuple[str, str]]): A list of tuples containing (user_id, message) for each message in the discussion.
            transparency (str): The visibility status of the discussion (e.g., public, private).
            project_id (Optional[str]): The ID of the project that this discussion is related to.

        Returns:
            str: The ID of the created discussion.
        """
        discussion = Discussion(
            discussion_id=str(uuid.uuid4()),
            data=DiscussionData(
                title=title,
                author_id=self.user_id,
                active_contributors=contributors,
                content=content,
                created_time=datetime.now().isoformat(),
                transparency=transparency,
                project_id=project_id
            )
        )
        try:
            discussion_handler.post_insert(discussion.convert_to_dict())
        except Exception as e:
            return {"status": "error", "message": str(e)}
        
        return {"status": "success", "data": {"discussion_id": discussion.discussion_id}}

    def delete_discussion(self, discussion_id: str) -> dict:
        """Deletes an existing discussion.

        Args:
            discussion_id (str): The ID of the discussion to delete.
        """
        if discussion_id in self.discussion_data:
            query_item = {"discussion_id": discussion_id, "author_id": self.user_id}
            discussion_handler.post_delete(query_item)
            return {"status": "success", "data": {"discussion_id": discussion_id}}
        else:
            return {"status": "error", "message": "Discussion not found"}

    def post_to_discussion(self, discussion_id: str, user_id: str, message: str) -> dict:
        """Posts a message to an existing discussion.

        Args:
            discussion_id (str): The ID of the discussion to post to.
            user_id (str): The ID of the user posting the message.
            message (str): The content of the message.
        """
        discussion = self.discussion_data.get(discussion_id) #type: ignore
        if discussion:
            discussion_obj = Discussion(
                discussion_id=discussion_id,
                data=DiscussionData(
                    title=discussion["title"],
                    author_id=discussion["author_id"],
                    active_contributors=discussion["active_contributors"],
                    content=discussion["content"],
                    created_time=discussion["created_time"],
                    transparency=discussion["transparency"],
                    project_id=discussion["project_id"]
                )
            )
        else:
            return {"error": "Discussion not found"}
        try:
            username = user_handler.get_single_doc({"user_id": user_id}).get("username", "Unknown User")
            if username not in discussion_obj.data.active_contributors:
                discussion_obj.data.active_contributors.append(username)
            discussion_obj.data.content.append((username, message))
        except Exception as e:
            return {"error": str(e)}

        return {"status": "success", "data": discussion_obj.convert_to_dict()}

    def delete_from_discussion(self, discussion_id, user_id, message):
        discussion = self.discussion_data.get(discussion_id) #type: ignore
        if discussion:
            try:
                username = user_handler.get_single_doc({"user_id": user_id}).get("username", "Unknown User")
                if (username, message) in discussion["content"]:
                    discussion["content"].remove((username, message))
                    discussion_handler.post_update({"discussion_id": discussion_id}, {"content": discussion["content"]})
                else:
                    return {"error": "Message not found in discussion"}
            except Exception as e:
                return {"error": str(e)}
        else:
            return {"error": "Discussion not found"}
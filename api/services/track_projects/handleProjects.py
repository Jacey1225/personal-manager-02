from pymongo import AsyncMongoClient
from api.validation.handleProjects import ValidateProjectHandler
from api.services.calendar.eventSetup import RequestSetup
from api.services.calendar.handleDateTimes import DateTimeHandler
from api.config.fetchMongo import MongoHandler
from api.schemas.projects import ProjectDetails
from api.schemas.model import EventOutput
from api.schemas.calendar import CalendarEvent
from api.config.plugins.enable_google_api import SyncGoogleEvents, SyncGoogleTasks
from api.config.plugins.enable_apple_api import SyncAppleEvents
from pydantic import BaseModel, Field
from typing import Optional
import uuid

validator = ValidateProjectHandler()

#MARK: Host Actions
class HostActions(RequestSetup):
    def __init__(self, 
                 calendar_event: CalendarEvent,
                 event_output: EventOutput,
                 user_id: str, 
                 calendar_service, 
                 user_handler: MongoHandler, 
                 project_handler: MongoHandler):
        self.user_handler = user_handler
        self.project_handler = project_handler
        self.user_data = None

        super().__init__(
            calendar_event, 
            event_output, 
            user_id, 
            calendar_service)

    @classmethod
    async def fetch(cls, 
                    calendar_event: CalendarEvent,
                    event_output: EventOutput,
                    user_id: str, 
                    calendar_service,
                    user_handler,
                    project_handler):
        """Fetches the necessary data for the host actions.

        Args:
            user_id (str): The ID of the user.
            user_handler (_type_): The user handler instance.
            project_handler (_type_): The project handler instance.
            calendar_event (CalendarEvent, optional): The calendar event instance. Defaults to CalendarEvent().

        Returns:
            HostActions: An instance of the HostActions class.
        """
        self = cls(
            event_output=event_output,
            calendar_event=calendar_event,
            user_id=user_id,
            calendar_service=calendar_service,
            user_handler=user_handler,
            project_handler=project_handler)
        self.user_data = await self.user_handler.get_single_doc({"user_id": self.user_id})
        return self

    async def global_delete(self):
        """Deletes all projects for the user.

        Raises:
            ValueError: If the user is not found.
        """
        user = await self.user_handler.get_single_doc({"user_id": self.user_id})
        if not user:
            raise ValueError("User not found")
        username = user["username"]
        email = user["email"]
        password = user["password"]
        if email == "jaceysimps@gmail.com" and username == "jaceysimpson" and password == "WeLoveDoggies16!":
            await self.project_handler.delete_all()

    async def fetch_user_id(self, email: str, username: str):
        """Fetches the user ID associated with the given email and username.

        Args:
            email (str): The email of the user.
            username (str): The username of the user.

        Returns:
            str: The user ID if found, otherwise None.
        """
        try:
            user = await self.user_handler.get_single_doc({"email": email, "username": username})
            return user["user_id"] if user else None
        except Exception as e:
            print(f"Error fetching user ID: {e}")
            print(f"Function: {self.fetch_user_id.__name__}")
            return None

    async def fetch_name_email(self, user_id: str):
        """Fetches the name and email associated with the given user ID.

        Args:
            user_id (str): The user ID of the user.

        Returns:
            tuple: A tuple containing the name and email if found, otherwise (None, None).
        """
        try:
            user = await self.user_handler.get_single_doc({"user_id": user_id})
            if user:
                return user["username"], user["email"]
        except Exception as e:
            print(f"Error fetching name and email: {e}")
            print(f"Function: {self.fetch_name_email.__name__}")
        return None, None

    def get_user_permission(self, project_id: str) -> str:
        """Gets the user's permission level for a specific project.

        Args:
            project_id (str): The ID of the project.

        Returns:
            str: The user's permission level ('view', 'edit', 'admin').
        """
        if self.user_data is not None:
            if "projects" in self.user_data and project_id in self.user_data["projects"]:
                return self.user_data["projects"][project_id][1]  # The permission is the second element in the tuple
        return "view"  # Default permission
    
    @validator.validate_project_args
    async def list_projects(self):
        """Lists all projects for the user.

        Returns:
            list[dict]: A list of projects associated with the user.
        """
        print(f"User Data: {self.user_data}")
        if self.user_data and "projects" not in self.user_data:
            return []
        
        try:
            projects = []
            if self.user_data is None:
                raise ValueError("User data not found")
            for project_id, _ in self.user_data["projects"].items():
                project = await self.project_handler.get_single_doc({"project_id": project_id})
                if project:
                    for i, member in enumerate(project.get("project_members", [])):
                        user_id = member
                        username, email = await self.fetch_name_email(user_id)
                        if user_id:
                            project["project_members"][i] = (username, email)

                    projects.append(project)
            return projects
        except Exception as e:
            print(f"Error listing projects: {e}")
            print(f"Function: {self.list_projects.__name__}")
            return []

    @validator.validate_user_data
    @validator.validate_project_args
    async def create_project(self, project_name: str,
                            project_likes: int,
                            project_transparency: bool,
                            project_members: Optional[list[tuple[str, str]]] = None,
                            organizations: Optional[list[str]] = None) -> ProjectDetails | None:
        """Creates a new project for the user.

        Args:
            project_name (str): The name of the project.
            project_likes (int): The number of likes for the project.
            project_transparency (bool): The transparency status of the project.
            project_members (list[tuple[str]]): A list of user IDs associated with the project.
        """
        self.user_data = await self.user_handler.get_single_doc({"user_id": self.user_id})
        self.project_id = str(uuid.uuid4())
        project = await self.project_handler.get_single_doc({"project_id": self.project_id})
        while project:
            self.project_id = str(uuid.uuid4())
            project = await self.project_handler.get_single_doc({"project_id": self.project_id})
            
        member_ids = []
        if project_members:
            for email, username in project_members:
                user_id = await self.fetch_user_id(email, username)
                if user_id:
                    member_ids.append(user_id)

        self.project_details = ProjectDetails(
            project_name=project_name,
            project_id=self.project_id,
            project_likes=project_likes,
            project_transparency=project_transparency,
            project_members=member_ids if member_ids else [self.user_id],
            organizations=organizations if organizations else []
        )
        try:
            query_item = {"user_id": self.user_data["user_id"]}
            new_data = self.project_details.model_dump()
            self.user_data["projects"][self.project_id] = (project_name, "admin")
            await self.user_handler.post_update(query_item, self.user_data)
            await self.project_handler.post_insert(new_data)
        except Exception as e:
            print(f"Error creating project: {e}")
            print(f"Function: {self.create_project.__name__}")
            raise
        if self.project_details:
            return self.project_details

    @validator.validate_project_existence
    async def delete_project(self, project_id: str) -> None:
        """Deletes an existing project for the user.

        Args:
            project_id (str): The ID of the project to delete.
        """
        # Check if user has admin permission
        user_permission = self.get_user_permission(project_id)
        if user_permission != "admin":
            raise ValueError("User does not have permission to delete this project")
        
        if not self.user_data:
            raise ValueError("User data not found")
        del self.user_data["projects"][project_id]
        if project_id in self.user_data.get("projects_liked", []):
            self.user_data["projects_liked"].remove(project_id)

        try:
            query_item = {"user_id": self.user_data["user_id"]}
            await self.user_handler.post_update(query_item, self.user_data)

            if await self.project_handler.get_single_doc({"project_id": project_id}):
                await self.project_handler.post_delete({"project_id": project_id})
        except Exception as e:
            print(f"Error deleting project: {e}")
            print(f"Function: {self.delete_project.__name__}")
            pass

    @validator.validate_project_existence
    async def rename_project(self, project_id: str, new_name: str) -> None:
        user_permission = self.get_user_permission(project_id)
        if user_permission != "admin":
            raise ValueError("User does not have permission to rename this project")
        
        try:
            if await self.project_handler.get_single_doc({"project_id": project_id}):
                await self.project_handler.post_update({"project_id": project_id}, {"project_name": new_name})
        except Exception as e:
            print(f"Error renaming project: {e}")
            print(f"Function: {self.rename_project.__name__}")
            pass

    @validator.validate_user_data
    @validator.validate_project_events
    async def fetch_project_events(self, project_id: str) -> list[dict]:
        """Fetches events associated with a specific project.

        Args:
            project_id (str): The ID of the project to fetch events for.

        Returns:
            list[dict]: A list of events associated with the project.
        """
        try:
            all_events = []
            project = await self.project_handler.get_single_doc({"project_id": project_id})
            if project:
                for user_id in project["project_members"]:
                    user = await self.user_handler.get_single_doc({"user_id": user_id})
                    if user:
                        request_setup = RequestSetup(
                            CalendarEvent(description=f"Lazi: {project_id}"), 
                            EventOutput(), 
                            user_id, 
                            self.calendar_service)
                        request_setup.fetch_events_list()
                        all_events.extend(request_setup.calendar_insights.scheduled_events)
                
                all_events = DateTimeHandler("").sort_datetimes(all_events)
                for event in all_events:
                    self.calendar_insights.project_events.append(event)
            
            return self.calendar_insights.project_events
        except Exception as e:
            print(f"Error fetching project events in {self.fetch_project_events.__name__}: {e}")
            return []

    async def edit_transparency(self, project_id: str, transparency: bool) -> None:
        """Edits the transparency status of an existing project.

        Args:
            project_id (str): The ID of the project to edit.
            transparency (bool): The new transparency status (True for public, False for private).
        """
        # Check if user has admin permission
        user_permission = self.get_user_permission(project_id)
        if user_permission != "admin":
            raise ValueError("User does not have permission to edit project transparency")

        try:
            project = await self.project_handler.get_single_doc({"project_id": project_id})
            if project:
                project["project_transparency"] = transparency
                query_item = {"project_id": project_id}
                await self.project_handler.post_update(query_item, project)
        except Exception as e:
            print(f"Error editing project transparency: {e}")
            print(f"Function: {self.edit_transparency.__name__}")
            pass

    async def edit_permissions(self, project_id: str, email: str, username: str, permission: str) -> bool:
        """Edit the permissions of a user for a specific project.

        Args:
            email (str): The email of the user whose permissions are to be edited.
            username (str): The username of the user whose permissions are to be edited.
            permission (str): The new permission level to be assigned to the user.

        Raises:
            ValueError: If the user does not have permission to edit.
            ValueError: If the user is not found.
            ValueError: If the permission level is invalid.

        Returns:
            bool: True if the permissions were successfully edited, False otherwise.
        """
        try:
            user_data = await self.user_handler.get_single_doc({"email": email, "username": username})
            if not user_data:
                raise ValueError("User not found")
            if permission not in ["view", "edit", "admin"]:
                raise ValueError("Invalid permission level")

            if user_data.get("projects", {}).get(project_id, [])[1] != "admin":
                raise ValueError("User does not have permission to edit")
            else:
                user_data["projects"][project_id][1] = permission
                await self.user_handler.post_update({"user_id": user_data["user_id"]}, user_data)
                return True

        except Exception as e:
            print(f"Error editing user permissions: {e}")
            print(f"Function: {self.edit_permissions.__name__}")
            return False


#MARK: Guest Actions
class GuestActions(HostActions):
    def __init__(self, 
                 calendar_event: CalendarEvent,
                 event_output: EventOutput,
                 calendar_service,
                 user_id: str, 
                 user_handler,
                 project_handler):
        super().__init__(
            calendar_event,
            event_output,
            user_id,
            calendar_service,
            user_handler, 
            project_handler)

    
    @validator.validate_project
    @validator.validate_project_args
    async def view_project(self, project_id: str) -> tuple[dict, dict]:
        """Fetches project details for a specific project.

        Args:
            project_id (str): The ID of the project to fetch.

        Returns:
            tuple: The project details and user data with permissions.
        """
        try:
            project = await self.project_handler.get_single_doc({"project_id": project_id})
            if not project:
                raise ValueError("Project not found")
            for i, user_id in enumerate(project.get("project_members", [])):
                username, email = await self.fetch_name_email(user_id)
                if username and email:
                    project["project_members"][i] = (email, username)
        except Exception as e:
            print(f"Error fetching project details: {e}")
            print(f"Function: {self.view_project.__name__}")
            return {}, {}

        try:
            permission = "view"  
            if self.user_data and "projects" in self.user_data and project_id in self.user_data["projects"]:
                permission = self.user_data["projects"][project_id][1]  

            if not self.user_data:
                raise ValueError("User data not found")
            user_info = {
                "user_id": self.user_data.get("user_id", ""),
                "email": self.user_data.get("email", ""),
                "username": self.user_data.get("username", ""),
                "projects": self.user_data.get("projects", {}),
                "projects_liked": self.user_data.get("projects_liked", []),
                "permission": permission
            }
            
            return project, user_info
        except Exception as e:
            print(f"Error fetching user data: {e}")
            print(f"Function: {self.view_project.__name__}")
            return project, {}

    @validator.validate_user_data
    async def like_project(self, project_id: str) -> None:
        """Likes an existing project.

        Args:
            project_id (str): The ID of the project to like.
        """
        try:
            project = await self.project_handler.get_single_doc({"project_id": project_id})
            if not self.user_data:
                raise ValueError("User data not found")
            if project:
                if project_id not in self.user_data.get("projects_liked", []):
                    self.user_data["projects_liked"].append(project_id)
                    query_item = {"user_id": self.user_data["user_id"]}
                    await self.user_handler.post_update(query_item, self.user_data)

                    project["project_likes"] = project.get("project_likes", 0) + 1
                    await self.project_handler.post_update({"project_id": project_id}, project)
        except Exception as e:
            print(f"Error liking project: {e}")
            print(f"Function: {self.like_project.__name__}")
            pass
            
    @validator.validate_user_data
    async def remove_like(self, project_id: str) -> None:
        """Removes a like from an existing project.

        Args:
            project_id (str): The ID of the project to remove a like from.
        """
        try:
            if not self.user_data:
                raise ValueError("User data not found")
            if project_id in self.user_data.get("projects_liked", []):
                self.user_data["projects_liked"].remove(project_id)
                query_item = {"user_id": self.user_data["user_id"]}
                await self.user_handler.post_update(query_item, self.user_data)
                project = await self.project_handler.get_single_doc({"project_id": project_id})
                if project:
                    project["project_likes"] = max(0, project.get("project_likes", 0) - 1)
                    await self.project_handler.post_update({"project_id": project_id}, project)
        except Exception as e:
            print(f"Error removing like: {e}")
            print(f"Function: {self.remove_like.__name__}")
            pass

    @validator.validate_project
    async def add_project_member(self, project_id: str, new_email: str, username: str, code: str) -> None:
        """Adds a new member to an existing project.

        Args:
            project_id (str): The ID of the project to add a member to.
            new_email (str): The email of the new member to add.
            username (str): The username of the new member to add.
            code (str): The code associated with the project to gain access.
        """
        user_permission = self.get_user_permission(project_id)
        if user_permission not in ["edit", "admin"]:
            raise ValueError("User does not have permission to add members to this project")
            
        try:
            project = await self.project_handler.get_single_doc({"project_id": project_id})
            if project:
                if (new_email, username) not in project["project_members"] and code == project.get("project_id"):
                    new_user = await self.user_handler.get_single_doc({"email": new_email, "username": username})
                    if new_user:
                        project["project_members"].append(new_user["user_id"])
                        new_user["projects"][project_id] = (project["project_name"], "view")
                        await self.user_handler.post_update({"user_id": new_user["user_id"]}, new_user)
                    query_item = {"project_id": project_id}
                    await self.project_handler.post_update(query_item, project)
        except Exception as e:
            print(f"Error adding project member: {e}")
            print(f"Function: {self.add_project_member.__name__}")

    @validator.validate_project
    async def delete_project_member(self, project_id: str, user_id: str) -> None:
        """Deletes a member from an existing project.

        Args:
            project_id (str): The ID of the project to delete a member from.
            user_id (str): The ID of the member to delete.
        """
        user_permission = self.get_user_permission(project_id)
        if user_permission not in ["edit", "admin"] and self.user_id != user_id:
            raise ValueError("User does not have permission to delete members from this project")
        
        try:
            project = await self.project_handler.get_single_doc({"project_id": project_id})
            user = await self.user_handler.get_single_doc({"user_id": user_id})
            if user and project:
                project["project_members"].remove(user["user_id"])
                user["projects"].remove(project_id)
                await self.project_handler.post_update({"project_id": project_id}, project)
                await self.user_handler.post_update({"user_id": user_id}, user)
        except Exception as e:
            print(f"Error deleting project member: {e}")
            print(f"Function: {self.delete_project_member.__name__}")
            pass
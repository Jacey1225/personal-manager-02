from src.validators.handleProjects import ValidateProjectHandler
from src.google_calendar.eventSetup import RequestSetup
from src.google_calendar.handleDateTimes import DateTimeHandler
from src.model_setup.structure_model_output import EventDetails
from pydantic import BaseModel, Field
from typing import Optional
import uuid
from src.fetchMongo import MongoHandler

user_handler = MongoHandler("userCredentials")
project_handler = MongoHandler("openProjects")
validator = ValidateProjectHandler()

class ProjectDetails(BaseModel):
    project_name: str = Field(..., description="Name of the project")
    project_id: str = Field(..., description="Unique identifier for the project")
    project_likes: int = Field(default=0, description="Number of likes for the project")
    project_transparency: bool = Field(default=True, description="Transparency status of the project(True: public - False: private)")
    project_members: list[str] = Field(..., description="List of user IDs associated with the project")
    organizations: Optional[list[str]] = Field(default=[], description="Organization IDs associated with the project")

#MARK: Host Actions
class HostActions(RequestSetup):
    def __init__(self, user_id, event_details: EventDetails = EventDetails()):
        self.event_details = event_details
        self.user_id = user_id
        self.user_data = user_handler.get_single_doc({"user_id": self.user_id})

        self.project_id = str(uuid.uuid4())
        while project_handler.get_single_doc({"project_id": self.project_id}):
            self.project_id = str(uuid.uuid4())

        super().__init__(self.event_details, user_id)
        self.all_events = []

    def global_delete(self):
        user = user_handler.get_single_doc({"user_id": self.user_id})
        if not user:
            raise ValueError("User not found")
        username = user["username"]
        email = user["email"]
        password = user["password"]
        if email == "jaceysimps@gmail.com" and username == "jaceysimpson" and password == "WeLoveDoggies16!":
            project_handler.delete_all()

    def fetch_user_id(self, email, username):
        """Fetches the user ID associated with the given email and username.

        Args:
            email (str): The email of the user.
            username (str): The username of the user.

        Returns:
            str: The user ID if found, otherwise None.
        """
        try:
            user = user_handler.get_single_doc({"email": email, "username": username})
            return user["user_id"] if user else None
        except Exception as e:
            print(f"Error fetching user ID: {e}")
            print(f"Function: {self.fetch_user_id.__name__}")
            return None

    def fetch_name_email(self, user_id):
        """Fetches the name and email associated with the given user ID.

        Args:
            user_id (str): The user ID of the user.

        Returns:
            tuple: A tuple containing the name and email if found, otherwise (None, None).
        """
        try:
            user = user_handler.get_single_doc({"user_id": user_id})
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
        if "projects" in self.user_data and project_id in self.user_data["projects"]:
            return self.user_data["projects"][project_id][1]  # The permission is the second element in the tuple
        return "view"  # Default permission

    @validator.validate_project_identifier 
    def tie_project(self) -> EventDetails: #Not part of the model API
        """Ties a project to the event details based on the project name.

        Args:
            event_details (EventDetails): The details of the event being processed.

        Returns:
            EventDetails: The updated event details with project information.
        """
        try:
            for key, (project_name, permission) in self.user_data["projects"].items():
                if permission == "view":
                    continue
                if project_name.lower() in self.event_details.input_text.lower():
                    self.event_details.transparency = "transparent"
                    self.event_details.guestsCanModify = True
                    self.event_details.description = f"Lazi: {key}"
                    break

            return self.event_details
        except Exception as e:
            print(f"Error tying project to event details: {e}")
            print(f"Function: {self.tie_project.__name__}")
            return self.event_details

    @validator.validate_project_args
    def list_projects(self):
        """Lists all projects for the user.

        Returns:
            list[dict]: A list of projects associated with the user.
        """
        print(f"User Data: {self.user_data}")
        if "projects" not in self.user_data:
            return []
        
        try:
            projects = []
            for project_id, _ in self.user_data["projects"].items():
                if project_handler.get_single_doc({"project_id": project_id}):
                    project = project_handler.get_single_doc({"project_id": project_id})
                    for i, member in enumerate(project.get("project_members", [])):
                        user_id = member
                        username, email = self.fetch_name_email(user_id)
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
    def create_project(self, project_name: str, project_likes: int, project_transparency: bool, 
                       project_members: Optional[list[tuple[str, str]]] = None, organizations: Optional[list[str]] = None) -> ProjectDetails | None:
        """Creates a new project for the user.

        Args:
            project_name (str): The name of the project.
            project_likes (int): The number of likes for the project.
            project_transparency (bool): The transparency status of the project.
            project_members (list[tuple[str]]): A list of user IDs associated with the project.
        """
        member_ids = []
        if project_members:
            for email, username in project_members:
                user_id = self.fetch_user_id(email, username)
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
            user_handler.post_update(query_item, self.user_data)
            project_handler.post_insert(new_data)
        except Exception as e:
            print(f"Error creating project: {e}")
            print(f"Function: {self.create_project.__name__}")
            raise
        if self.project_details:
            return self.project_details

    @validator.validate_project_existence
    def delete_project(self, project_id: str) -> None:
        """Deletes an existing project for the user.

        Args:
            project_id (str): The ID of the project to delete.
        """
        # Check if user has admin permission
        user_permission = self.get_user_permission(project_id)
        if user_permission != "admin":
            raise ValueError("User does not have permission to delete this project")
            
        del self.user_data["projects"][project_id]
        if project_id in self.user_data.get("projects_liked", []):
            self.user_data["projects_liked"].remove(project_id)

        try:
            query_item = {"user_id": self.user_data["user_id"]}
            user_handler.post_update(query_item, self.user_data)

            if project_handler.get_single_doc({"project_id": project_id}):
                project_handler.post_delete({"project_id": project_id})
        except Exception as e:
            print(f"Error deleting project: {e}")
            print(f"Function: {self.delete_project.__name__}")
            pass

    @validator.validate_project_existence
    def rename_project(self, project_id: str, new_name: str) -> None:
        user_permission = self.get_user_permission(project_id)
        if user_permission != "admin":
            raise ValueError("User does not have permission to rename this project")
        
        try:
            if project_handler.get_single_doc({"project_id": project_id}):
                project_handler.post_update({"project_id": project_id}, {"project_name": new_name})
        except Exception as e:
            print(f"Error renaming project: {e}")
            print(f"Function: {self.rename_project.__name__}")
            pass

    @validator.validate_user_data
    @validator.validate_project_events
    def fetch_project_events(self, project_id: str) -> list[dict]:
        """Fetches events associated with a specific project.

        Args:
            project_id (str): The ID of the project to fetch events for.

        Returns:
            list[dict]: A list of events associated with the project.
        """
        try:
            if project_handler.get_single_doc({"project_id": project_id}):
                project = project_handler.get_single_doc({"project_id": project_id})

                for user_id in project["project_members"]:
                    user = user_handler.get_single_doc({"user_id": user_id})
                    if user:
                        request_setup = RequestSetup(EventDetails(), user_id, description=f"Lazi: {project_id}")
                        request_setup.fetch_events_list()
                        self.all_events.extend(request_setup.calendar_insights.scheduled_events)
                
                self.all_events = DateTimeHandler("").sort_datetimes(self.all_events)
                for event in self.all_events:
                    self.calendar_insights.project_events.append(event.model_dump())
            
            return self.calendar_insights.project_events
        except Exception as e:
            print(f"Error fetching project events in {self.fetch_project_events.__name__}: {e}")
            return []

    def edit_transparency(self, project_id: str, transparency: bool) -> None:
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
            if project_handler.get_single_doc({"project_id": project_id}):
                project = project_handler.get_single_doc({"project_id": project_id})
                project["project_transparency"] = transparency
                query_item = {"project_id": project_id}
                project_handler.post_update(query_item, project)
        except Exception as e:
            print(f"Error editing project transparency: {e}")
            print(f"Function: {self.edit_transparency.__name__}")
            pass

    def edit_permissions(self, project_id: str, email: str, username: str, permission: str) -> bool:
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
            user_data = user_handler.get_single_doc({"email": email, "username": username})
            if not user_data:
                raise ValueError("User not found")

            if permission not in ["view", "edit", "admin"]:
                raise ValueError("Invalid permission level")

            if self.user_data.get("projects", {}).get(project_id, [])[1] != "admin":
                raise ValueError("User does not have permission to edit")
            else:
                user_data["projects"][project_id][1] = permission
                user_handler.post_update({"user_id": user_data["user_id"]}, user_data)
                return True

        except Exception as e:
            print(f"Error editing user permissions: {e}")
            print(f"Function: {self.edit_permissions.__name__}")
            return False


#MARK: Guest Actions
class GuestActions(HostActions):
    def __init__(self, user_id: str, event_details: EventDetails = EventDetails()):
        self.user_id = user_id
        self.event_details = event_details
        super().__init__(self.user_id, self.event_details)
        self.user_data = user_handler.get_single_doc({"user_id": self.user_id})

    @validator.validate_project
    @validator.validate_project_args
    def view_project(self, project_id: str) -> tuple[dict, dict]:
        """Fetches project details for a specific project.

        Args:
            project_id (str): The ID of the project to fetch.

        Returns:
            tuple: The project details and user data with permissions.
        """
        try:
            project = project_handler.get_single_doc({"project_id": project_id})
            if not project:
                raise ValueError("Project not found")
            for i, user_id in enumerate(project.get("project_members", [])):
                username, email = self.fetch_name_email(user_id)
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
    def like_project(self, project_id: str) -> None:
        """Likes an existing project.

        Args:
            project_id (str): The ID of the project to like.
        """
        try:
            if project_handler.get_single_doc({"project_id": project_id}):
                project = project_handler.get_single_doc({"project_id": project_id})
                if project:
                    if project_id not in self.user_data.get("projects_liked", []):
                        self.user_data["projects_liked"].append(project_id)
                        query_item = {"user_id": self.user_data["user_id"]}
                        user_handler.post_update(query_item, self.user_data)

                        project["project_likes"] = project.get("project_likes", 0) + 1
                        project_handler.post_update({"project_id": project_id}, project)
        except Exception as e:
            print(f"Error liking project: {e}")
            print(f"Function: {self.like_project.__name__}")
            pass

        return self.user_data["projects_liked"]
            
    @validator.validate_user_data
    def remove_like(self, project_id: str) -> None:
        """Removes a like from an existing project.

        Args:
            project_id (str): The ID of the project to remove a like from.
        """
        try:
            if project_id in self.user_data.get("projects_liked", []):
                self.user_data["projects_liked"].remove(project_id)
                query_item = {"user_id": self.user_data["user_id"]}
                user_handler.post_update(query_item, self.user_data)
                if project_handler.get_single_doc({"project_id": project_id}):
                    project = project_handler.get_single_doc({"project_id": project_id})
                    if project:
                        project["project_likes"] = max(0, project.get("project_likes", 0) - 1)
                        project_handler.post_update({"project_id": project_id}, project)
        except Exception as e:
            print(f"Error removing like: {e}")
            print(f"Function: {self.remove_like.__name__}")
            pass

        return self.user_data["projects_liked"]
    
    @validator.validate_project
    def add_project_member(self, project_id: str, new_email: str, username: str, code: str) -> None:
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
            if project_handler.get_single_doc({"project_id": project_id}):
                project = project_handler.get_single_doc({"project_id": project_id})
                if (new_email, username) not in project["project_members"] and code == project.get("project_id"):
                    new_user = user_handler.get_single_doc({"email": new_email, "username": username})
                    if new_user:
                        project["project_members"].append(new_user["user_id"])
                        new_user["projects"][project_id] = (project["project_name"], "view")
                        user_handler.post_update({"user_id": new_user["user_id"]}, new_user)
                    query_item = {"project_id": project_id}
                    project_handler.post_update(query_item, project)
        except Exception as e:
            print(f"Error adding project member: {e}")
            print(f"Function: {self.add_project_member.__name__}")

    @validator.validate_project
    def delete_project_member(self, project_id: str, user_id: str) -> None:
        """Deletes a member from an existing project.

        Args:
            project_id (str): The ID of the project to delete a member from.
            user_id (str): The ID of the member to delete.
        """
        user_permission = self.get_user_permission(project_id)
        if user_permission not in ["edit", "admin"] and self.user_id != user_id:
            raise ValueError("User does not have permission to delete members from this project")
        
        try:
            if project_handler.get_single_doc({"project_id": project_id}):
                project = project_handler.get_single_doc({"project_id": project_id})
                user = user_handler.get_single_doc({"user_id": user_id})
                if user and project:
                    project["project_members"].remove(user["user_id"])
                    user["projects"].remove(project_id)
                    project_handler.post_update({"project_id": project_id}, project)
                    user_handler.post_update({"user_id": user_id}, user)
        except Exception as e:
            print(f"Error deleting project member: {e}")
            print(f"Function: {self.delete_project_member.__name__}")
            pass
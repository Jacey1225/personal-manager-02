from src.validators.validators import ValidateProjectHandler
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
    project_members: list[tuple[str, str]] = Field(..., description="List of user emails and usernames associated with the project")
    organizations: Optional[list[str]] = Field(default=None, description="Organization IDs associated with the project")

#MARK: Host Actions
class HostActions(RequestSetup):
    def __init__(self, user_id, event_details: EventDetails = EventDetails()):
        self.event_details = event_details
        super().__init__(self.event_details, user_id)
        self.user_id = user_id
        self.user_data = user_handler.get_single_doc({"user_id": self.user_id})

        self.project_id = str(uuid.uuid4())

        self.all_events = self.calendar_insights.scheduled_events

    @validator.validate_project_identifier 
    def tie_project(self) -> EventDetails: #Not part of the model API
        """Ties a project to the event details based on the project name.

        Args:
            event_details (EventDetails): The details of the event being processed.

        Returns:
            EventDetails: The updated event details with project information.
        """
        for key, project_name in self.user_data["projects"].items():
            if project_name.lower() in self.event_details.input_text.lower():
                self.event_details.transparency = "transparent"
                self.event_details.guestsCanModify = True
                self.event_details.description = f"Lazi: {key}"
                break

        return self.event_details

    def list_projects(self):
        """Lists all projects for the user.

        Returns:
            list[dict]: A list of projects associated with the user.
        """
        print(f"User Data: {self.user_data}")
        if "projects" not in self.user_data:
            return []
        
        projects = []
        for project_id, _ in self.user_data["projects"].items():
            print(f"Fetching project with ID: {project_id}")
            if project_handler.get_single_doc({"project_id": project_id}):
                project = project_handler.get_single_doc({"project_id": project_id})
                projects.append(project)
        return projects

    @validator.validate_user_data
    def create_project(self, project_name: str, project_likes: int, project_transparency: bool, project_members: Optional[list[tuple[str, str]]] = None, organizations: Optional[list[str]] = None) -> None:
        """Creates a new project for the user.

        Args:
            project_name (str): The name of the project.
            project_likes (int): The number of likes for the project.
            project_transparency (bool): The transparency status of the project.
            project_members (list[tuple[str, str]]): A list of user emails and usernames associated with the project.
        """
        self.project_details = ProjectDetails(
            project_name=project_name,
            project_id=self.project_id,
            project_likes=project_likes,
            project_transparency=project_transparency,
            project_members=project_members if project_members else [(self.personal_email, "")],
            organizations=organizations if organizations else []
        )

        query_item = {"user_id": self.user_data["user_id"]}
        new_data = self.project_details.model_dump()
        self.user_data["projects"][self.project_id] = project_name
        user_handler.post_update(query_item, self.user_data)
        project_handler.post_insert(new_data)

    @validator.validate_project_existence
    def delete_project(self, project_id: str) -> None:
        """Deletes an existing project for the user.

        Args:
            project_id (str): The ID of the project to delete.
        """
        del self.user_data["projects"][project_id]
        if project_id in self.user_data.get("projects_liked", []):
            self.user_data["projects_liked"].remove(project_id)

        query_item = {"user_id": self.user_data["user_id"]}
        user_handler.post_update(query_item, self.user_data)

        if project_handler.get_single_doc({"project_id": project_id}):
            project_handler.post_delete({"project_id": project_id})

    @validator.validate_project_existence
    def rename_project(self, project_id: str, new_name: str) -> None:
        if project_handler.get_single_doc({"project_id": project_id}):
            project_handler.post_update({"project_id": project_id}, {"project_name": new_name})

    @validator.validate_user_data
    @validator.validate_project_events
    def fetch_project_events(self, project_id: str) -> list[dict]:
        """Fetches events associated with a specific project.

        Args:
            project_id (str): The ID of the project to fetch events for.

        Returns:
            list[dict]: A list of events associated with the project.
        """
        if project_handler.get_single_doc({"project_id": project_id}):
            project = project_handler.get_single_doc({"project_id": project_id})

            for email, username in project["project_members"]:
                if user_handler.get_single_doc({"email": email, "username": username}):
                    user_id = user_handler.get_single_doc({"email": email, "username": username}).get("user_id")
                    if user_id:
                        request_setup = RequestSetup(EventDetails(), user_id)
                        self.all_events.extend(request_setup.calendar_insights.scheduled_events)

            self.all_events = DateTimeHandler("").sort_datetimes(self.all_events)
            for event in self.all_events:
                if event.description == f"Lazi: {project_id}":
                    self.calendar_insights.project_events.append(event.model_dump())
                    print(f"Event added for project {project_id}: {event.model_dump()}")
        
        return self.calendar_insights.project_events
    
    @validator.validate_project
    def add_project_member(self, project_id: str, new_email: str, username: str) -> None:
        """Adds a new member to an existing project.

        Args:
            project_id (str): The ID of the project to add a member to.
            new_email (str): The email of the new member to add.
            username (str): The username of the new member to add.
        """
        if project_handler.get_single_doc({"project_id": project_id}):
            project = project_handler.get_single_doc({"project_id": project_id})
            if (new_email, username) not in project["project_members"]:
                project["project_members"].append((new_email, username))

            query_item = {"project_id": project_id}
            user_handler.post_update(query_item, project)

    @validator.validate_project
    def delete_project_member(self, project_id: str, email: str, username: str) -> None:
        """Deletes a member from an existing project.

        Args:
            project_id (str): The ID of the project to delete a member from.
            email (str): The email of the member to delete.
        """
        if project_handler.get_single_doc({"project_id": project_id}):
            project = project_handler.get_single_doc({"project_id": project_id})
            if (email, username) in project["project_members"]:
                project["project_members"].remove((email, username))
                query_item = {"project_id": project_id}
                user_handler.post_update(query_item, project)

    def edit_transparency(self, project_id: str, transparency: bool) -> None:
        """Edits the transparency status of an existing project.

        Args:
            project_id (str): The ID of the project to edit.
            transparency (bool): The new transparency status (True for public, False for private).
        """
        if project_handler.get_single_doc({"project_id": project_id}):
            project = project_handler.get_single_doc({"project_id": project_id})
            project["project_transparency"] = transparency
            query_item = {"project_id": project_id}
            user_handler.post_update(query_item, project)

#MARK: Guest Actions
class GuestActions(RequestSetup):
    def __init__(self, user_id: str, event_details: EventDetails = EventDetails()):
        self.user_id = user_id
        self.event_details = event_details
        super().__init__(self.event_details, self.user_id)
        self.user_data = user_handler.get_single_doc({"user_id": self.user_id})

    @validator.validate_project
    def view_project(self, project_id: str) -> tuple[dict, dict]:
        """Fetches project details for a specific project.

        Args:
            project_id (str): The ID of the project to fetch.

        Returns:
            dict: The details of the requested project.
        """
        if project_handler.get_single_doc({"project_id": project_id}):
            project = project_handler.get_single_doc({"project_id": project_id})
            print(f"Viewing project: {project}")
            return project, self.user_data
        return {}, self.user_data

    @validator.validate_user_data
    def like_project(self, project_id: str) -> None:
        """Likes an existing project.

        Args:
            project_id (str): The ID of the project to like.
        """
        if project_handler.get_single_doc({"project_id": project_id}):
            project = project_handler.get_single_doc({"project_id": project_id})
            if project:
                if project_id not in self.user_data.get("projects_liked", []):
                    self.user_data["projects_liked"].append(project_id)
                    query_item = {"user_id": self.user_data["user_id"]}
                    user_handler.post_update(query_item, self.user_data)

                    project["project_likes"] = project.get("project_likes", 0) + 1
                    project_handler.post_update({"project_id": project_id}, project)
        return self.user_data["projects_liked"]
            
    @validator.validate_user_data
    def remove_like(self, project_id: str) -> None:
        """Removes a like from an existing project.

        Args:
            project_id (str): The ID of the project to remove a like from.
        """
        if project_id in self.user_data.get("projects_liked", []):
            self.user_data["projects_liked"].remove(project_id)
            query_item = {"user_id": self.user_data["user_id"]}
            user_handler.post_update(query_item, self.user_data)
            if project_handler.get_single_doc({"project_id": project_id}):
                project = project_handler.get_single_doc({"project_id": project_id})
                if project:
                    project["project_likes"] = max(0, project.get("project_likes", 0) - 1)
                    project_handler.post_update({"project_id": project_id}, project)

        return self.user_data["projects_liked"]
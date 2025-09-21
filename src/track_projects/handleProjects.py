from src.validators.validators import ValidateProjectHandler
from src.google_calendar.eventSetup import RequestSetup
from src.google_calendar.handleDateTimes import DateTimeHandler
from src.model_setup.structure_model_output import EventDetails
import os
import json
from pydantic import BaseModel, Field
import uuid
from src.fetchMongo import MongoHandler

mongo_client = MongoHandler()
validator = ValidateProjectHandler()

class ProjectDetails(BaseModel):
    project_name: str = Field(..., description="Name of the project")
    project_id: str = Field(..., description="Unique identifier for the project")
    project_members: list[tuple[str, str]] = Field(..., description="List of user emails and usernames associated with the project")

class FetchProject(RequestSetup):
    def __init__(self, user_id, event_details: EventDetails = EventDetails()):
        self.event_details = event_details
        super().__init__(self.event_details, user_id)
        self.host_user = user_id
        self.user_data = mongo_client.get_single_doc({"user_id": self.host_user})
        self.project_id = str(uuid.uuid4())

        self.all_events = self.calendar_insights.scheduled_events

    def list_projects(self):
        """Lists all projects for the user.

        Returns:
            list[dict]: A list of projects associated with the user.
        """
        if "projects" not in self.user_data:
            return []

        return [project for key, project in self.user_data["projects"].items()]

    def create_project(self, project_name: str, project_members: list[tuple[str, str]]) -> None:
        """Creates a new project for the user.

        Args:
            project_name (str): The name of the project.
            project_members (list[tuple[str, str]]): A list of user emails and usernames associated with the project.
        """
        self.project_details = ProjectDetails(
            project_name=project_name,
            project_id=self.project_id,
            project_members=project_members
        )
        
        if "projects" not in self.user_data:
            self.user_data["projects"] = {}

        query_item = {"user_id": self.user_data["user_id"]}
        new_data = self.project_details.model_dump()
        self.user_data["projects"][self.project_id] = new_data
        mongo_client.post_update(query_item, self.user_data)

    @validator.validate_project_existence
    def delete_project(self, project_id: str) -> None:
        """Deletes an existing project for the user.

        Args:
            project_id (str): The ID of the project to delete.
        """
        del self.user_data["projects"][project_id]

        query_item = {"user_id": self.user_data["user_id"]}
        mongo_client.post_update(query_item, self.user_data)

    @validator.validate_project_existence
    def rename_project(self, project_id: str, new_name: str) -> None:
        self.user_data["projects"][project_id]["project_name"] = new_name
        query_item = {"user_id": self.user_data["user_id"]}
        mongo_client.post_update(query_item, self.user_data)

    @validator.validate_project_identifier
    def tie_project(self) -> EventDetails:
        """Ties a project to the event details based on the project name.

        Args:
            event_details (EventDetails): The details of the event being processed.

        Returns:
            EventDetails: The updated event details with project information.
        """
        for key, project in self.user_data["projects"].items():
            if project["project_name"].lower() in self.event_details.input_text.lower():
                self.event_details.transparency = "transparent"
                self.event_details.guestsCanModify = True
                self.event_details.description = f"Lazi: {key}"
                break

        return self.event_details

    @validator.validate_project_events
    def fetch_project_events(self, project_id: str) -> list[dict]:
        """Fetches events associated with a specific project.

        Args:
            project_id (str): The ID of the project to fetch events for.

        Returns:
            list[dict]: A list of events associated with the project.
        """
        project = ProjectDetails(
            project_name=self.user_data["projects"][project_id]["project_name"],
            project_id=project_id,
            project_members=self.user_data["projects"][project_id]["project_members"]
        )

        for email, username in project.project_members:
            if mongo_client.get_single_doc({"email": email, "username": username}):
                user_id = mongo_client.get_single_doc({"email": email, "username": username}).get("user_id")
                if user_id:
                    request_setup = RequestSetup(EventDetails(), user_id)
                    self.all_events.extend(request_setup.calendar_insights.scheduled_events)

        for event in self.all_events:
            if event.description == f"Lazi: {project_id}":
                self.calendar_insights.project_events.append(event.model_dump())
                print(f"Event added for project {project_id}: {event.model_dump()}")
        return self.calendar_insights.project_events
    
    def add_project_member(self, project_id: str, new_email: str, username: str) -> None:
        """Adds a new member to an existing project.

        Args:
            project_id (str): The ID of the project to add a member to.
            new_email (str): The email of the new member to add.
            username (str): The username of the new member to add.
        """
        # Fix: Use project_members instead of project_emails
        if (new_email, username) not in self.user_data["projects"][project_id]["project_members"]:
            self.user_data["projects"][project_id]["project_members"].append((new_email, username))

            query_item = {"user_id": self.user_data["user_id"]}
            mongo_client.post_update(query_item, self.user_data)

    def delete_project_member(self, project_id: str, email: str, username: str) -> None:
        """Deletes a member from an existing project.

        Args:
            project_id (str): The ID of the project to delete a member from.
            email (str): The email of the member to delete.
        """
        if (email, username) in self.user_data["projects"][project_id]["project_members"]:
            self.user_data["projects"][project_id]["project_members"].remove((email, username))

            query_item = {"user_id": self.user_data["user_id"]}
            mongo_client.post_update(query_item, self.user_data)

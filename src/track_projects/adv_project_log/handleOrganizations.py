from src.google_calendar.eventSetup import RequestSetup
from src.model_setup.structure_model_output import EventDetails
from src.fetchMongo import MongoHandler
from pydantic import BaseModel, Field
from typing import Optional
from src.track_projects.handleProjects import ProjectDetails
import uuid

user_handler = MongoHandler("userCredentials")
organization_handler = MongoHandler("organizations")
project_handler = MongoHandler("projects")

class Organization(BaseModel):
    name: str = Field(description="The name of the organization")
    members: list[tuple[str, str]] = Field(description="List of user emails and usernames associated with the organization")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the organization")
    projects: list[str] = Field(default_factory=list, description="List of project IDs associated with the organization")

class HandleOrganizations(RequestSetup):
    def __init__(self, user_id: str, event_details: EventDetails = EventDetails()):
        self.user_id = user_id
        self.event_details = event_details
        super().__init__(self.event_details, self.user_id)

        self.user_data = user_handler.get_single_doc({"user_id": self.user_id})
        self.organization_id = str(uuid.uuid4())
        while organization_handler.get_single_doc({"id": self.organization_id}):
            self.organization_id = str(uuid.uuid4())

    def create_organization(self, name: str, members: list[tuple[str, str]], projects: list[str]) -> str:
        """Creates a new organization.

        Args:
            name (str): The name of the organization.
            members (list[tuple[str, str]]): A list of user emails and usernames associated with the organization.
            projects (list[str]): A list of project IDs associated with the organization.

        Returns:
            str: The ID of the newly created organization.
        """
        organization = Organization(name=name, members=members, projects=projects)
        organization_id = organization.id
        
        organization_handler.post_insert(organization.model_dump())
        return organization_id
    
    def delete_organization(self, organization_id: str) -> bool:
        """Deletes an existing organization.

        Args:
            organization_id (str): The ID of the organization to delete.

        Returns:
            bool: True if the organization was deleted, False otherwise.
        """
        if organization_handler.get_single_doc({"id": organization_id}):
            organization_handler.post_delete({"id": organization_id})
            return True
        return False

    def list_organizations(self) -> list[Organization]:
        """Lists all organizations.

        Returns:
            list[Organization]: A list of all organizations.
        """
        organizations = organization_handler.get_all() or {}

        user_organizations = []
        for organization in organizations:
            if self.user_id in organization.get("members", []):
                user_organizations.append(Organization(**organization))
        
        return user_organizations

    def add_member(self, organization_id: str, new_email: str, new_username: str) -> bool:
        """Adds a new member to an organization.

        Args:
            organization_id (str): The ID of the organization.
            new_email (str): The email of the new member.
            new_username (str): The username of the new member.

        Returns:
            bool: True if the member was added, False otherwise.
        """
        organization_data = organization_handler.get_single_doc({"id": organization_id})
        if organization_data:
            members = organization_data.get("members", [])
            if (new_email, new_username) not in members:
                members.append((new_email, new_username))
                organization_handler.post_update({"id": organization_id}, {"members": members})
                return True
        return False

    def delete_member(self, organization_id: str, email: str) -> bool:
        """Deletes a member from an organization.

        Args:
            organization_id (str): The ID of the organization.
            email (str): The email of the member to delete.

        Returns:
            bool: True if the member was deleted, False otherwise.
        """
        organization_data = organization_handler.get_single_doc({"id": organization_id})
        if organization_data:
            members = organization_data.get("members", [])
            for member in members:
                if member[0] == email:
                    members.remove(member)
                    organization_handler.post_update({"id": organization_id}, {"members": members})
                    return True
        return False
    
    def add_project(self, organization_id: str, project_details: ProjectDetails) -> bool:
        """Adds a new project to an organization.

        Args:
            organization_id (str): The ID of the organization.
            project_details (ProjectDetails): The details of the project to add.

        Returns:
            bool: True if the project was added, False otherwise.
        """
        organization_data = organization_handler.get_single_doc({"id": organization_id})
        if organization_data:
            projects = organization_data.get("projects", [])
            if project_details.project_id not in projects:
                projects.append(project_details.project_id)
                organization_handler.post_update({"id": organization_id}, {"projects": projects})
                project_handler.post_insert(project_details.model_dump())
                
                return True
        return False

    def delete_project(self, organization_id: str, project_id: str) -> bool:
        """Deletes a project from an organization.

        Args:
            organization_id (str): The ID of the organization.
            project_id (str): The ID of the project to delete.

        Returns:
            bool: True if the project was deleted, False otherwise.
        """
        organization_data = organization_handler.get_single_doc({"id": organization_id})
        if organization_data:
            projects = organization_data.get("projects", [])
            if project_id in projects:
                projects.remove(project_id)
                organization_handler.post_update({"id": organization_id}, {"projects": projects})
                project_handler.post_delete({"project_id": project_id})
                return True
        return False
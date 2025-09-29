from src.google_calendar.eventSetup import RequestSetup
from src.model_setup.structure_model_output import EventDetails
from src.fetchMongo import MongoHandler
from src.validators.handleOrganizations import ValidateOrganizations
from pydantic import BaseModel, Field
from src.track_projects.handleProjects import ProjectDetails
import uuid

user_handler = MongoHandler("userCredentials")
organization_handler = MongoHandler("organizations")
project_handler = MongoHandler("projects")

validator = ValidateOrganizations()

class Organization(BaseModel):
    name: str = Field(description="The name of the organization")
    members: list[str] = Field(description="List of user IDs associated with the organization")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the organization")
    projects: list[str] = Field(default_factory=list, description="List of project IDs associated with the organization")

class HandleOrganizations():
    def __init__(self, user_id: str):
        self.user_id = user_id

        self.user_data = user_handler.get_single_doc({"user_id": self.user_id})

        if not self.user_data.get("organizations", []):
            self.user_data["organizations"] = []

        self.organization_id = str(uuid.uuid4())
        while organization_handler.get_single_doc({"id": self.organization_id}):
            self.organization_id = str(uuid.uuid4())

    def create_organization(self, name: str, members: list[str], projects: list[str]) -> str:
        """Creates a new organization.

        Args:
            name (str): The name of the organization.
            members (list[str]): A list of user IDs associated with the organization.
            projects (list[str]): A list of project IDs associated with the organization.

        Returns:
            str: The ID of the newly created organization.
        """
        organization = Organization(name=name, members=members, projects=projects)
        organization_id = organization.id
        
        organization_handler.post_insert(organization.model_dump())

        self.user_data["organizations"].append(organization_id)
        user_handler.post_update({"user_id": self.user_id}, self.user_data)
        return organization_id
    
    @validator.validate_organization_data
    def delete_organization(self, organization_id: str) -> bool:
        """Deletes an existing organization.

        Args:
            organization_id (str): The ID of the organization to delete.

        Returns:
            bool: True if the organization was deleted, False otherwise.
        """
        if organization_handler.get_single_doc({"id": organization_id}):
            organization_handler.post_delete({"id": organization_id})

            self.user_data["organizations"].remove(organization_id)
            user_handler.post_update({"user_id": self.user_id}, self.user_data)
            return True
        return False

    def list_organizations(self) -> list[Organization]:
        """Lists all organizations.

        Returns:
            list[Organization]: A list of all organizations.
        """
        organizations = self.user_data["organizations"]

        user_organizations = []
        for organization_id in organizations:
            org_data = organization_handler.get_single_doc({"id": organization_id})
            for i, member_id in enumerate(org_data.get("members", [])):
                member_info = user_handler.get_single_doc({"user_id": member_id})
                if member_info:
                    org_data["members"][i] = member_info.get("username", member_id)    
            if org_data:
                user_organizations.append(Organization(**org_data))

        return user_organizations

    @validator.validate_organization_data
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
            new_member = user_handler.get_single_doc({"email": new_email, "username": new_username})
            if new_member and new_member["user_id"] not in members:
                members.append(new_member["user_id"])
                organization_handler.post_update({"id": organization_id}, {"members": members})

                new_member.get("organizations", []).append(organization_id)
                user_handler.post_update({"user_id": new_member["user_id"]}, new_member)
                return True
        return False

    @validator.validate_organization_data
    def delete_member(self, organization_id: str, user_id: str) -> bool:
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
                if member == user_id:
                    try:
                        members.remove(member)
                        organization_handler.post_update({"id": organization_id}, {"members": members})

                        member.get("organizations", []).remove(organization_id)
                        user_handler.post_update({"user_id": member["user_id"]}, member)
                    except Exception as e:
                        print(f"Error deleting member: {e}")
                        return False
                    return True
        return False
    
    @validator.validate_organization_data
    def add_project(self, organization_id: str, project_details: ProjectDetails) -> bool:
        """Adds a new project to an organization.

        Args:
            organization_id (str): The ID of the organization.
            project_details (ProjectDetails): The details of the project to add.

        Returns:
            bool: True if the project was added, False otherwise.
        """
        organization_data = organization_handler.get_single_doc({"id": organization_id})
        if not project_details.organizations:
            project_details.organizations = []
        if organization_data:
            try:

                organization_data.get("projects", []).append(project_details.project_id) if project_details.project_id not in organization_data.get("projects", []) else None
                organization_handler.post_update({"id": organization_id}, organization_data)

                project_details.organizations.append(organization_id)
                project_handler.post_update({"project_id": project_details.project_id}, project_details.model_dump())

                print(f"Successfully added project {project_details.project_id} to organization {organization_id}")
            except Exception as e:
                print(f"Error adding project: {e}")
                return False
            return True
        print(f"Cannot find organization from {organization_id} or project from {project_details.project_id}")
        return False

    @validator.validate_organization_data
    def delete_project(self, organization_id: str, project_id: str) -> bool:
        """Deletes a project from an organization.

        Args:
            organization_id (str): The ID of the organization.
            project_id (str): The ID of the project to delete.

        Returns:
            bool: True if the project was deleted, False otherwise.
        """
        organization_data = organization_handler.get_single_doc({"id": organization_id})
        project_data = project_handler.get_single_doc({"project_id": project_id})
        if organization_data and project_data:
            try:
                organization_data.get("projects", []).remove(project_id) if project_id in organization_data.get("projects", []) else None
                organization_handler.post_update({"id": organization_id}, organization_data)

                project_data.get("organizations", []).remove(organization_id) if organization_id in project_data.get("organizations", []) else None
                project_handler.post_update({"project_id": project_data["project_id"]}, project_data)
            except Exception as e:
                print(f"Error deleting project: {e}")
                return False
            return True
        return False
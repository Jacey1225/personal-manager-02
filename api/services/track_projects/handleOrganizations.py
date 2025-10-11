from api.config.fetchMongo import MongoHandler
from api.validation.handleOrganizations import ValidateOrganizations
from pydantic import BaseModel, Field
from api.services.track_projects.handleProjects import ProjectDetails
from api.schemas.projects import Organization
import uuid 

validator = ValidateOrganizations()

class HandleOrganizations():
    def __init__(self, user_id: str, user_handler, organization_handler, project_handler):
        self.user_id = user_id
        self.user_handler = user_handler
        self.organization_handler = organization_handler
        self.project_handler = project_handler

    @classmethod
    async def fetch(cls, 
                    user_id: str,
                    user_handler,
                    organization_handler,
                    project_handler):
        self = cls(user_id, user_handler, organization_handler, project_handler)
        self.user_data = await self.user_handler.get_single_doc({"user_id": self.user_id})

        if not self.user_data.get("organizations", []):
            self.user_data["organizations"] = []
        return self

    async def create_organization(self, name: str, members: list[str], projects: list[str]) -> str:
        """Creates a new organization.

        Args:
            name (str): The name of the organization.
            members (list[str]): A list of user IDs associated with the organization.
            projects (list[str]): A list of project IDs associated with the organization.

        Returns:
            str: The ID of the newly created organization.
        """
        organization = Organization(name=name, members=members, projects=projects)
        organization = await self.organization_handler.get_single_doc({"id": self.organization_id})
        organization_id = organization.id
        while organization:
            organization_id = str(uuid.uuid4())
            organization = await self.organization_handler.get_single_doc({"id": organization_id})

        await self.organization_handler.post_insert(organization.model_dump())

        self.user_data["organizations"].append(organization_id)
        await self.user_handler.post_update({"user_id": self.user_id}, self.user_data)
        return organization_id
    
    @validator.validate_organization_data
    async def delete_organization(self, organization_id: str) -> bool:
        """Deletes an existing organization.

        Args:
            organization_id (str): The ID of the organization to delete.

        Returns:
            bool: True if the organization was deleted, False otherwise.
        """
        if await self.organization_handler.get_single_doc({"id": organization_id}):
            await self.organization_handler.post_delete({"id": organization_id})

            self.user_data["organizations"].remove(organization_id)
            await self.user_handler.post_update({"user_id": self.user_id}, self.user_data)
            return True
        return False

    async def list_organizations(self) -> list[Organization]:
        """Lists all organizations.

        Returns:
            list[Organization]: A list of all organizations.
        """
        organizations = self.user_data["organizations"]

        user_organizations = []
        for organization_id in organizations:
            org_data = await self.organization_handler.get_single_doc({"id": organization_id})
            for i, member_id in enumerate(org_data.get("members", [])):
                member_info = await self.user_handler.get_single_doc({"user_id": member_id})
                if member_info:
                    org_data["members"][i] = member_info.get("username", member_id)    
            if org_data:
                user_organizations.append(Organization(**org_data))

        return user_organizations

    @validator.validate_organization_data
    async def add_member(self, organization_id: str, new_email: str, new_username: str) -> bool:
        """Adds a new member to an organization.

        Args:
            organization_id (str): The ID of the organization.
            new_email (str): The email of the new member.
            new_username (str): The username of the new member.

        Returns:
            bool: True if the member was added, False otherwise.
        """
        organization_data = await self.organization_handler.get_single_doc({"id": organization_id})
        if organization_data:
            members = organization_data.get("members", [])
            new_member = await self.user_handler.get_single_doc({"email": new_email, "username": new_username})
            if new_member and new_member["user_id"] not in members:
                members.append(new_member["user_id"])
                await self.organization_handler.post_update({"id": organization_id}, {"members": members})

                new_member.get("organizations", []).append(organization_id)
                await self.user_handler.post_update({"user_id": new_member["user_id"]}, new_member)
                return True
        return False

    @validator.validate_organization_data
    async def delete_member(self, organization_id: str, user_id: str) -> bool:
        """Deletes a member from an organization.

        Args:
            organization_id (str): The ID of the organization.
            email (str): The email of the member to delete.

        Returns:
            bool: True if the member was deleted, False otherwise.
        """
        organization_data = await self.organization_handler.get_single_doc({"id": organization_id})
        if organization_data:
            members = organization_data.get("members", [])
            for member in members:
                if member == user_id:
                    try:
                        members.remove(member)
                        await self.organization_handler.post_update({"id": organization_id}, {"members": members})

                        member.get("organizations", []).remove(organization_id)
                        await self.user_handler.post_update({"user_id": member["user_id"]}, member)
                    except Exception as e:
                        print(f"Error deleting member: {e}")
                        return False
                    return True
        return False
    
    @validator.validate_organization_data
    async def add_project(self, organization_id: str, project_details: ProjectDetails) -> bool:
        """Adds a new project to an organization.

        Args:
            organization_id (str): The ID of the organization.
            project_details (ProjectDetails): The details of the project to add.

        Returns:
            bool: True if the project was added, False otherwise.
        """
        organization_data = await self.organization_handler.get_single_doc({"id": organization_id})
        if not project_details.organizations:
            project_details.organizations = []
        if organization_data:
            try:

                organization_data.get("projects", []).append(project_details.project_id) if project_details.project_id not in organization_data.get("projects", []) else None
                await self.organization_handler.post_update({"id": organization_id}, organization_data)

                project_details.organizations.append(organization_id)
                await self.project_handler.post_update({"project_id": project_details.project_id}, project_details.model_dump())

                print(f"Successfully added project {project_details.project_id} to organization {organization_id}")
            except Exception as e:
                print(f"Error adding project: {e}")
                return False
            return True
        print(f"Cannot find organization from {organization_id} or project from {project_details.project_id}")
        return False

    @validator.validate_organization_data
    async def delete_project(self, organization_id: str, project_id: str) -> bool:
        """Deletes a project from an organization.

        Args:
            organization_id (str): The ID of the organization.
            project_id (str): The ID of the project to delete.

        Returns:
            bool: True if the project was deleted, False otherwise.
        """
        organization_data = await self.organization_handler.get_single_doc({"id": organization_id})
        project_data = await self.project_handler.get_single_doc({"project_id": project_id})
        if organization_data and project_data:
            try:
                organization_data.get("projects", []).remove(project_id) if project_id in organization_data.get("projects", []) else None
                await self.organization_handler.post_update({"id": organization_id}, organization_data)

                project_data.get("organizations", []).remove(organization_id) if organization_id in project_data.get("organizations", []) else None
                await self.project_handler.post_update({"project_id": project_data["project_id"]}, project_data)
            except Exception as e:
                print(f"Error deleting project: {e}")
                return False
            return True
        return False
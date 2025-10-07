from api.services.track_projects.handleOrganizations import HandleOrganizations, ProjectDetails
from api.schemas.projects import CreateOrgRequest, OrgRequest, ProjectDetails


class OrganizationModel:
    @staticmethod
    async def create_organization(request: CreateOrgRequest, user_id: str):
        handler = HandleOrganizations(user_id)
        return handler.create_organization(
            name=request.name,
            members=request.members,
            projects=request.projects
        )

    @staticmethod
    async def delete_organization(request: OrgRequest):
        handler = HandleOrganizations(request.user_id)
        return handler.delete_organization(request.organization_id)

    @staticmethod
    async def list_organizations(user_id: str):
        handler = HandleOrganizations(user_id)
        return handler.list_organizations()
    
    @staticmethod
    async def add_member(request: OrgRequest, new_email: str, new_username: str):
        handler = HandleOrganizations(request.user_id)
        return handler.add_member(request.organization_id, new_email, new_username)

    @staticmethod
    async def remove_member(request: OrgRequest, email: str):
        handler = HandleOrganizations(request.user_id)
        return handler.delete_member(request.organization_id, email)

    @staticmethod
    async def add_project(request: OrgRequest, project_dict: dict):
        handler = HandleOrganizations(request.user_id)
        project_details = ProjectDetails(**project_dict)
        return handler.add_project(request.organization_id, project_details)

    @staticmethod
    async def remove_project(request: OrgRequest, project_id: str):
        handler = HandleOrganizations(request.user_id)
        return handler.delete_project(request.organization_id, project_id)
from api.services.track_projects.handleOrganizations import HandleOrganizations, ProjectDetails
from api.schemas.projects import CreateOrgRequest, OrgRequest, ProjectDetails
from api.config.fetchMongo import MongoHandler
from api.config.cache import organization_cache, async_cached

user_config = MongoHandler("userAuthDatabase", "userCredentials")
organization_config = MongoHandler("userAuthDatabase", "openOrganizations")
project_config = MongoHandler("userAuthDatabase", "openProjects")

class OrganizationModel:
    @staticmethod
    async def create_organization(request: CreateOrgRequest, user_id: str):
        handler = await HandleOrganizations.fetch(user_id, user_config, organization_config, project_config)
        return await handler.create_organization(
            name=request.name,
            members=request.members,
            projects=request.projects
        )

    @staticmethod
    async def delete_organization(request: OrgRequest):
        handler = await HandleOrganizations.fetch(request.user_id, user_config, organization_config, project_config)
        return await handler.delete_organization(request.organization_id)

    @staticmethod
    @async_cached(cache=organization_cache)
    async def list_organizations(user_id: str):
        handler = await HandleOrganizations.fetch(user_id, user_config, organization_config, project_config)
        return await handler.list_organizations()
    
    @staticmethod
    async def add_member(request: OrgRequest, new_email: str, new_username: str):
        user_id = request.user_id
        handler = HandleOrganizations(user_id, user_config, organization_config, project_config)
        return await handler.add_member(request.organization_id, new_email, new_username)

    @staticmethod
    async def remove_member(request: OrgRequest, email: str):
        user_id = request.user_id
        handler = await HandleOrganizations.fetch(user_id, user_config, organization_config, project_config)
        return await handler.delete_member(request.organization_id, email)

    @staticmethod
    async def add_project(request: OrgRequest, project_dict: dict):
        user_id = request.user_id
        handler = await HandleOrganizations.fetch(user_id, user_config, organization_config, project_config)
        project_details = ProjectDetails(**project_dict)
        return handler.add_project(request.organization_id, project_details)

    @staticmethod
    async def remove_project(request: OrgRequest, project_id: str):
        user_id = request.user_id
        handler = await HandleOrganizations.fetch(user_id, user_config, organization_config, project_config)
        return await handler.delete_project(request.organization_id, project_id)
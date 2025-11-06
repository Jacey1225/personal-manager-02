from fastapi import APIRouter, Query
from api.resources.organization_model import OrganizationModel
from api.schemas.projects import CreateOrgRequest, OrgRequest
from api.config.cache import organization_cache

organization_router = APIRouter()
commander = OrganizationModel()

@organization_router.post("/organizations/create_org")
async def create_organization(request: CreateOrgRequest, user_id: str = Query(...)):
    return await commander.create_organization(request, user_id=user_id)

@organization_router.delete("/organizations/delete_org", response_model=OrgRequest)
async def delete_organization(request: OrgRequest):
    return await commander.delete_organization(request)

@organization_router.get("/organizations/list_orgs")
async def list_organizations(request: OrgRequest):
    cache_key = organization_cache.get_cache_key(
        "list_organizations",
        (request.user_id,),
        {"force_refresh": request.force_refresh})
    if request.force_refresh:
        org_data = await organization_cache.get_or_set(
            cache_key,
            commander.list_organizations,
            user_id=request.user_id
        )
        if request.user_id in org_data:
            await organization_cache.pop(cache_key)
    return await commander.list_organizations(user_id=request.user_id)

@organization_router.post("/organizations/add_member")
async def add_member(request: OrgRequest, new_email: str, new_username: str):
    return await commander.add_member(request, new_email=new_email, new_username=new_username)

@organization_router.delete("/organizations/remove_member")
async def remove_member(request: OrgRequest, email: str):
    return await commander.remove_member(request, email=email)

@organization_router.post("/organizations/add_project")
async def add_project(request: OrgRequest, project_dict: dict):
    return await commander.add_project(request, project_dict=project_dict)

@organization_router.delete("/organizations/remove_project")
async def remove_project(request: OrgRequest, project_id: str):
    return await commander.remove_project(request, project_id=project_id)
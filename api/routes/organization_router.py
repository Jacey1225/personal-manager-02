from fastapi import APIRouter, Query
from api.build.organization_model import OrganizationModel
from api.schemas.projects import CreateOrgRequest, OrgRequest

organization_router = APIRouter()
commander = OrganizationModel()

@organization_router.post("/organizations/create_org")
async def create_organization(request: CreateOrgRequest, user_id: str = Query(...)):
    return await commander.create_organization(request, user_id=user_id)

@organization_router.delete("/organizations/delete_org", response_model=OrgRequest)
async def delete_organization(request: OrgRequest):
    return await commander.delete_organization(request)

@organization_router.get("/organizations/list_orgs")
async def list_organizations(user_id: str = Query(...)):
    return await commander.list_organizations(user_id=user_id)

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
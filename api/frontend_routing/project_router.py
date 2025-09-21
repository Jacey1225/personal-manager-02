from fastapi import APIRouter
from api.commandline.project_model import ProjectModel

project_router = APIRouter()
commander = ProjectModel()


@project_router.post("/projects/create_project")
async def create_project(request):
    return await commander.create_project(request)

@project_router.post("/projects/delete_project")
async def delete_project(request):
    return await commander.delete_project(request)

@project_router.post("/projects/rename_project")
async def rename_project(request):
    return await commander.rename_project(request)

@project_router.get("/projects/events/{project_id}")
async def get_project_events(project_id: str, user_id: str):
    return await commander.get_project_events(project_id, user_id)

@project_router.get("/projects/add_member")
async def add_project_member(project_id: str, user_id: str, new_email: str, new_username: str):
    return await commander.add_project_member(project_id, user_id, new_email, new_username)

@project_router.get("/projects/delete_member")
async def delete_project_member(project_id: str, user_id: str, email: str, username: str):
    return await commander.delete_project_member(project_id, user_id, email, username)

@project_router.get("/projects/list")
async def list_projects(user_id: str):
    return await commander.list_projects(user_id)


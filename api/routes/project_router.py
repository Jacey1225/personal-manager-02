from fastapi import APIRouter, Query
from api.resources.project_model import ProjectModel
from api.schemas.projects import CreateProjectRequest, ModifyProjectRequest
from api.config.cache import project_cache

project_router = APIRouter()
commander = ProjectModel()

#MARK: Guest Actions

@project_router.get("/projects/view_project")
async def view_project(project_id: str = Query(...), user_id: str = Query(...), project_name: str = Query(...), force_refresh: bool = Query(False)):
    request = ModifyProjectRequest(project_id=project_id, user_id=user_id, project_name=project_name, force_refresh=force_refresh)
    cache_key = project_cache.get_cache_key(
        "view_project",
        (project_id, user_id, project_name), 
        {"force_refresh": force_refresh})
    if force_refresh:
        project_data = await project_cache.get_or_set(
            cache_key,
            commander.view_project,
            request)
        if request in project_data:
            await project_cache.pop(cache_key)
    return await commander.view_project(request)

@project_router.post("/projects/like_project")
async def like_project(request: ModifyProjectRequest):
    return await commander.like_project(request)

@project_router.post("/projects/unlike_project")
async def unlike_project(request: ModifyProjectRequest):
    return await commander.remove_like(request)

#MARK: Host Actions

@project_router.post("/projects/global_delete")
async def global_delete(request: ModifyProjectRequest):
    return await commander.global_delete(request)

@project_router.post("/projects/create_project")
async def create_project(request: CreateProjectRequest):
    return await commander.create_project(request)

@project_router.post("/projects/delete_project")
async def delete_project(request: ModifyProjectRequest):
    return await commander.delete_project(request)

@project_router.post("/projects/rename_project")
async def rename_project(request: ModifyProjectRequest):
    return await commander.rename_project(request)

@project_router.get("/projects/events/{project_id}")
async def get_project_events(project_id: str, user_id: str = Query(...), force_refresh: bool = Query(False)):
    request = ModifyProjectRequest(project_id=project_id, user_id=user_id, project_name="", force_refresh=force_refresh)
    cache_key = project_cache.get_cache_key(
        "get_project_events",
        (project_id, user_id), 
        {"force_refresh": force_refresh})
    if force_refresh:
        project_data = await project_cache.get_or_set(
            cache_key,
            commander.get_project_events,
            request)    
        if request in project_data:
            await project_cache.pop(cache_key)
        
    return await commander.get_project_events(request)

@project_router.get("/projects/add_member")
async def add_project_member(project_id: str = Query(...), user_id: str = Query(...), new_email: str = Query(...), new_username: str = Query(...), code: str = Query(...)):
    request = ModifyProjectRequest(project_id=project_id, user_id=user_id, project_name="")
    return await commander.add_project_member(request, new_email, new_username, code)

@project_router.delete("/projects/delete_member")
async def delete_project_member(project_id: str = Query(...), user_id: str = Query(...), email: str = Query(...), username: str = Query(...)):
    request = ModifyProjectRequest(project_id=project_id, user_id=user_id, project_name="")
    return await commander.delete_project_member(request, email, username)

@project_router.get("/projects/list")
async def list_projects(user_id: str = Query(...)):
    return await commander.list_projects(user_id=user_id)

@project_router.post("/projects/edit_transparency")
async def edit_project_transparency(request: ModifyProjectRequest, transparency: bool):
    return await commander.edit_transparency(request, transparency)

@project_router.post("/projects/edit_permission")
async def edit_project_permission(request: ModifyProjectRequest, email: str, username: str, new_permission: str):
    return await commander.edit_permission(request, email, username, new_permission)

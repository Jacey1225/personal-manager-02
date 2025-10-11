from fastapi import APIRouter, Query
from api.build.discussion_model import DiscussionsModel
from api.config.fetchMongo import MongoHandler
from api.schemas.projects import DiscussionRequest, DiscussionData

discussion_router = APIRouter()
commander = DiscussionsModel()

@discussion_router.get("/discussions/view_discussion")
async def view_discussion(user_id: str = Query(...), project_id: str = Query(...), discussion_id: str = Query(...)):
    request = DiscussionRequest(user_id=user_id, project_id=project_id)
    return await commander.view_discussion(request, discussion_id)

@discussion_router.get("/discussions/list_project_discussions")
async def list_project_discussions(user_id: str = Query(...), project_id: str = Query(...)):
    request = DiscussionRequest(user_id=user_id, project_id=project_id)
    return await commander.list_project_discussions(request)

@discussion_router.post("/discussions/create_discussion")
async def create_discussion(request: DiscussionRequest, discussion_data: DiscussionData):
    return await commander.create_discussion(request, discussion_data)

@discussion_router.post("/discussions/delete_discussion")
async def delete_discussion(request: DiscussionRequest, discussion_id: str = Query(...)):
    return await commander.delete_discussion(request, discussion_id)

@discussion_router.post("/discussions/add_member")
async def add_member_to_discussion(request: DiscussionRequest, discussion_id: str = Query(...)):
    return await commander.add_member_to_discussion(request, discussion_id)

@discussion_router.post("/discussions/remove_member")
async def remove_member_from_discussion(request: DiscussionRequest, discussion_id: str = Query(...)):
    return await commander.remove_member_from_discussion(request, discussion_id)

@discussion_router.post("/discussions/post_message")
async def post_message(request: DiscussionRequest, discussion_id: str = Query(...), message: str = Query(...)):
    return await commander.post_to_discussion(request, discussion_id, message)

@discussion_router.post("/discussions/remove_message")
async def remove_message_from_discussion(request: DiscussionRequest, discussion_id: str = Query(...), message: str = Query(...)):
    return await commander.delete_from_discussion(request, discussion_id, message)

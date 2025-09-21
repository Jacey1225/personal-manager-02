from api.commandline.coordination_model import CoordinationModel
from fastapi import APIRouter

router = APIRouter()
commander = CoordinationModel()

@router.post("/coordinate/fetch_users")
async def fetch_users(request_body: dict):
    return await commander.fetch_users(request_body)

@router.post("/coordinate/get_availability")
async def get_availability(request_body: dict):
    return await commander.get_availability(request_body)
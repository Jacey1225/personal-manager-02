from fastapi import APIRouter
from typing import Dict, Any
from api.schemas.model import InputRequest
from api.resources.main_model import MainModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

event_router = APIRouter()
commander = MainModel()

@event_router.post("/scheduler/process_input")
async def process_input(input_request: InputRequest) -> list[Dict[str, Any]]:
    events = await commander.generate_events(input_request)
    return await commander.process_input(input_request.user_id, events)

@event_router.post("/scheduler/delete_event/{event_id}")
async def delete_event(event_id: str, request_body: dict) -> dict:
    return await commander.delete_event(event_id, request_body)

@event_router.post("/scheduler/update_event/{event_id}")
async def update_event(event_id: str, request_body: dict) -> dict:
    return await commander.update_event(event_id, request_body)

@event_router.get("/")
def read_root():
    print("Root endpoint accessed")
    return {"Hello": "Jacey"}
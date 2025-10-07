from fastapi import APIRouter
from typing import Dict

event_router = APIRouter()

@event_router.post("/scheduler/fetch_events")
async def fetch_events(input_request: InputRequest) -> list[dict]:
    return await commander.fetch_events(input_request)

@event_router.post("/scheduler/process_input")
async def process_input(events: list[dict]) -> list[Dict[str, Any]]:
    return await commander.process_input(events)

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
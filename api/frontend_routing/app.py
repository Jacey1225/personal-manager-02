from fastapi import FastAPI
from api.frontend_routing.auth_router import auth_router
from api.frontend_routing.tasklist_router import task_list_router
from api.frontend_routing.project_router import project_router
from api.frontend_routing.coordination_router import coordination_router
from api.frontend_routing.discussion_router import discussion_router

from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from src.model_setup.structure_model_output import EventDetails
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
from datetime import datetime, timezone
import pytz
from api.commandline.main_model import MainModel


app = FastAPI() 
scheduler = AsyncIOScheduler()
commander = MainModel()

# More restrictive CORS for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
app.include_router(auth_router)
app.include_router(task_list_router)
app.include_router(project_router)
app.include_router(coordination_router)
app.include_router(discussion_router)

class InputRequest(BaseModel):
    input_text: str
    user_id: str

class EventRequest(BaseModel):
    event_details: EventDetails
    user_id: str

responseRequest = {
    "user_id": str,
    "status": str,
    "message": str,
    "event_requested": dict,
    "calendar_insights": Optional[dict]
}

async def midnight_refresh():
    utc_now = datetime.now(timezone.utc)
    local_tz = pytz.timezone('America/Los_Angeles')  # Change to your local timezone
    local_now = utc_now.astimezone(local_tz)
    print(f"Midnight refresh executed at {local_now.strftime('%Y-%m-%d %H:%M:%S %Z%z')}")

@app.post("/scheduler/fetch_events")
async def fetch_events(input_request: InputRequest) -> list[dict]:
    return await commander.fetch_events(input_request)

@app.post("/scheduler/process_input")
async def process_input(events: list[dict]) -> list[Dict[str, Any]]:
    return await commander.process_input(events)

@app.post("/scheduler/delete_event/{event_id}")
async def delete_event(event_id: str, request_body: dict) -> dict:
    return await commander.delete_event(event_id, request_body)

@app.post("/scheduler/update_event/{event_id}")
async def update_event(event_id: str, request_body: dict) -> dict:
    return await commander.update_event(event_id, request_body)


@app.get("/")
def read_root():
    return {"Hello": "Jacey"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
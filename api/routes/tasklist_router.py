from fastapi import APIRouter
from api.resources.tasklist_model import TaskListModel
from api.schemas.calendar import EventsRequest

task_list_router = APIRouter()
commander = TaskListModel()


@task_list_router.post("/task_list/list_events")
async def list_events(event_request: EventsRequest) -> list[dict]:
    print(f"Received event_request: {event_request}")
    return await commander.list_events(event_request)


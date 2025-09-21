from fastapi import APIRouter
from api.commandline.tasklist_model import TaskListModel, EventRequest

task_list_router = APIRouter()
commander = TaskListModel()


@task_list_router.post("/task_list/list_events")
async def list_events(event_request: EventRequest) -> list[dict]:
    print(f"Received event_request: {event_request}")
    return await commander.list_events(event_request)
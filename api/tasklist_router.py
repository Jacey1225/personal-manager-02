from fastapi import APIRouter, HTTPException
from src.google_calendar.eventSetup import RequestSetup
from src.model_setup.structure_model_output import EventDetails
from pydantic import BaseModel
from src.google_calendar.handleDateTimes import DateTimeHandler
# Define EventRequest locally to avoid circular import
class EventRequest(BaseModel):
    event_details: EventDetails
    user_id: str

task_list_router = APIRouter()


@task_list_router.post("/task_list/list_events")
async def list_events(event_request: EventRequest) -> list[dict]:
    """List events for a user.

    Args:
        event_request (EventRequest): The request object containing event details and user ID.

    Returns:
        list[dict]: A list of events for the user.
            event_details = event_request.event_details
            event_details.event_name = "None"
            event_details.action = "None"
            user_id = event_request.user_id
            request_setup = RequestSetup(event_details, user_id)
            events_list = request_setup.calendar_insights.scheduled_events
            events_json = []
            for event in events_list:
                formatted_times = format_datetimes(event.start, event.end if event.end else None)
                event_dict = {
                    "event_name": event.event_name,
                    "start_time": formatted_times["start_time"],
                    "end_time": formatted_times["end_time"],
                    "event_id": event.event_id
                }
                events_json.append(event_dict)
    """
    event_details = event_request.event_details
    event_details.event_name = "None"
    event_details.action = "None"
    user_id = event_request.user_id
    request_setup = RequestSetup(event_details, user_id)
    events_list = request_setup.calendar_insights.scheduled_events
    events_json = []
    datetime_handler = DateTimeHandler(input_text="None")
    for event in events_list:
        formatted_times = datetime_handler.format_datetimes(event.start, event.end if event.end else None)
        event_dict = {
            "event_name": event.event_name,
            "start_time": formatted_times["start_time"],
            "end_time": formatted_times["end_time"],
            "event_id": event.event_id
        }
        events_json.append(event_dict)
    return events_json
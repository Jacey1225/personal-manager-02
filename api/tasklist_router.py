from fastapi import APIRouter, HTTPException
from src.google_calendar.eventSetup import RequestSetup
from datetime import datetime
from api.app import EventRequest
from typing import Optional

router = APIRouter()


@router.get("scheduler/list_events")
async def list_events(event_request: EventRequest) -> list[dict]:
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
    return events_json

def format_datetimes(event_start: datetime, event_end: Optional[datetime]) -> dict:
    start_formatted = event_start.strftime("%A, %B %d, %Y %I:%M %p")
    if event_end:
        end_formatted = event_end.strftime("%A, %B %d, %Y %I:%M %p")
    else:
        end_formatted = None
    return {
        "start_time": start_formatted,
        "end_time": end_formatted
    }
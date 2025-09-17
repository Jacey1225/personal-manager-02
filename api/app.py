from fastapi import FastAPI, HTTPException
from api.auth_router import auth_router
from api.tasklist_router import task_list_router
from api.project_router import project_router
from api.coordination_router import router as coordination_router
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from src.model_setup.structure_model_output import EventDetails, HandleResponse
from src.google_calendar.handleDateTimes import DateTimeSet
from src.google_calendar.handleEvents import CalendarInsights, AddToCalendar, DeleteFromCalendar, UpdateFromCalendar
from src.track_projects.handleProjects import FetchProject
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
from datetime import datetime, timezone
import pytz


app = FastAPI() 
scheduler = AsyncIOScheduler()

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

def process_response(status: str, message: str, event_requested: dict, calendar_insights: dict = {}) -> dict:
    """Processes the response for the user request.

    Args:
        status (str): The status of the response (e.g., "success", "error").
        message (str): A message providing additional information about the response.
        event_requested (dict): The event details that were requested by the user.
        calendar_insights (dict, optional): Insights from the calendar regarding the event. Defaults to None.

    Returns:
        dict: The processed response containing the status, message, event details, and any calendar insights.
    """
    response = {
        "status": status,
        "message": message,
        "event_requested": event_requested,
        "calendar_insights": calendar_insights
    }
    return response

@app.post("/scheduler/fetch_events")
async def fetch_events(input_request: InputRequest) -> list[dict]:
    """Fetch events from the calendar based on user input.

    Args:
        input_request (InputRequest): The user input request containing the input text.

    Returns:
        list[dict]: A list of dictionaries containing the fetched events.
    """
    input_text = input_request.input_text
    user_id = input_request.user_id

    response_handler = HandleResponse(input_text)
    events = response_handler.process_response() 
    list_events_dict = []
    for event in events:
        event_dict = {
            "user_id": user_id,
            "input_text": input_text,
            "event_name": event.event_name,
            "target_dates": event.datetime_obj.target_datetimes,
            "action": event.action,
            "response": event.response,
            "transparency": event.transparency,
            "guestsCanModify": event.guestsCanModify,
            "description": event.description
        }
        list_events_dict.append(event_dict)
    return list_events_dict


@app.post("/scheduler/process_input")
async def process_input(events: list[dict]) -> list[Dict[str, Any]]:
    """Processes user input, generates, a response, and returns that response as a list of dictionaries for either further processing or user feedback.

    Args:
        input_request (InputRequest): The user input request containing the input text.

    Raises:
        HTTPException: If an error occurs while processing the input request.

    Returns:
        list[Dict[str, Any]]: A list of dictionaries containing the response for each processed event. This is needed if the system needs to request additional information
    """
    try:
        requests = []
        for event in events:
            user_id = event['user_id']
            event_details = EventDetails(
                input_text=event['input_text'],
                event_name=event['event_name'],
                datetime_obj=DateTimeSet(target_datetimes=event['target_dates']),
                action=event['action'],
                response=event['response'],
                transparency=event['transparency'],
                guestsCanModify=event['guestsCanModify'],
                description=event['description']
            )
            event_details = FetchProject(user_id, event_details).tie_project()

            print(f"Event Details: {event_details}")
            if event_details.action.lower() == "add":
                add_handler = AddToCalendar(event_details, user_id)
                add_handler.add_event()
                result = add_handler.event_details.response

                responseRequest = process_response("completed", result, event)
                requests.append(responseRequest)

            elif event_details.action.lower() == "delete":
                delete_handler = DeleteFromCalendar(event_details, user_id)
                delete_handler.find_matching_events()
                calendar_insights_dict = {
                    "matching_events": delete_handler.calendar_insights.matching_events,
                    "is_event": delete_handler.calendar_insights.is_event   
                }

                responseRequest = process_response("request event ID", f"Please select an event to delete as {event_details.event_name.upper()}", event, calendar_insights_dict)
                requests.append(responseRequest)

            elif event_details.action.lower() == "update":
                update_handler = UpdateFromCalendar(event_details, user_id)
                update_handler.find_matching_events()
                calendar_insights_dict = {
                    "matching_events": update_handler.calendar_insights.matching_events,
                    "is_event": update_handler.calendar_insights.is_event
                }
                event_details.datetime_obj.target_datetimes = [(start, end) for start, end in update_handler.event_details.datetime_obj.target_datetimes]
                responseRequest = process_response("request event ID", f"Please select an event to update as {event_details.event_name.upper()}", event, calendar_insights_dict)
                requests.append(responseRequest)

            else:
                result = f"Unknown action '{event_details.action}' for event '{event_details.event_name}'."
                responseRequest = process_response("failed", result, event)
                requests.append(responseRequest)

        return requests
        
    except Exception as e:
        print(f"I'm sorry, something went wrong. Please try again: {e}")
        return [{"status": "failed", "message": f"Something went wrong please try again."}]

@app.post("/scheduler/delete_event/{event_id}")
async def delete_event(event_id: str, request_body: dict) -> dict:
    """Delete an event from the calendar.

    Args:
        event_id (str): The ID of the event to delete.
        request_body (dict): The request body containing event details and calendar insights.

    Raises:
        HTTPException: If an error occurs while deleting the event.

    Returns:
        dict: A dictionary containing the status and message of the delete operation.
    """
    try:
        event_requested = request_body.get("event_requested", {})

        datetime_set = DateTimeSet(
            target_datetimes=event_requested.get("target_dates", [])
        )
        
        event_details = EventDetails(
            event_name=event_requested.get("event_name", "None"),
            datetime_obj=datetime_set,
            action=event_requested.get("action", "None"),
            response=event_requested.get("response", "None")
        )
        
        if event_details.event_name == "None" or not event_details.event_name:
            raise ValueError("Event name must be provided.")
        
        user_id = request_body.get("user_id", "None")
        calendar_insights_dict = request_body.get("calendar_insights", {})        
        calendar_insights = CalendarInsights(
            matching_events=calendar_insights_dict.get("matching_events", []),
            is_event=calendar_insights_dict.get("is_event", False)
        )

        delete_handler = DeleteFromCalendar(event_details, user_id)
        delete_handler.calendar_insights = calendar_insights

        delete_handler.delete_event(event_id)  
        return {"status": "success", "message": event_details.response}
    except Exception as e:
        print(f"I'm sorry, something went wrong. Please try again: {e}")
        return {"status": "failed", "message": f"Something went wrong please try again."}
    
def convert_to_iso(og_target_dates: list[tuple[str, Optional[str]]]) -> list[tuple[str, Optional[str]]]:
    """Convert original target dates to ISO format.

    Args:
        og_target_dates (list[tuple[str, Optional[str]]]): Original target dates.

    Returns:
        list[tuple[str, Optional[str]]]: Converted target dates in ISO format.
    """
    pdt = pytz.timezone('America/Los_Angeles')
    iso_dates = []
    for start, end in og_target_dates:
        try:
            start_dt = datetime.strptime(start, "%A, %B %d, %Y %I:%M %p")
            start_pdt = pdt.localize(start_dt).isoformat()
            if end:
                end_dt = datetime.strptime(end, "%A, %B %d, %Y %I:%M %p")
                end_pdt = pdt.localize(end_dt).isoformat()
            else:
                end_pdt = None
            print(f"Converted {start} and {end} to PDT: {start_pdt}, {end_pdt}")
        except Exception as e:
            print(f"Datetimes do not conform to format: {e}")
            start_pdt = start
            end_pdt = end

        iso_dates.append((start_pdt, end_pdt))
    return iso_dates

@app.post("/scheduler/update_event/{event_id}")
async def update_event(event_id: str, request_body: dict) -> dict:
    """Update an existing event in the calendar.

    Args:
        event_id (str): ID of the event to update.
        request_body (dict): The updated event details.
            {
                "event_name": "Updated Event",
                "target_dates": ["2023-10-01T10:00:00Z", "2023-10-01T11:00:00Z"],
                "action": "update",
                "response": "Event updated successfully"
            }

    Raises:
        HTTPException: If an error occurs while updating the event.

    Returns:
        dict: A dictionary containing the status and message of the update operation.
    """
    try:
        event_requested = request_body.get("event_requested", {})
        target_dates = convert_to_iso(event_requested.get("target_dates", []))

        datetime_set = DateTimeSet(
            target_datetimes=target_dates
        )
        
        event_details = EventDetails(
            event_name=event_requested.get("event_name", "None"),
            datetime_obj=datetime_set,
            action=event_requested.get("action", "None"),
            response=event_requested.get("response", "None")
        )
        print(f"Event Details Target Datetimes: {event_details.datetime_obj.target_datetimes}")

        calendar_insights_dict = request_body.get("calendar_insights", {})
        user_id = request_body.get("user_id", "None")
        calendar_insights = CalendarInsights(**calendar_insights_dict)

        update_handler = UpdateFromCalendar(event_details, user_id)
        update_handler.calendar_insights = calendar_insights
        update_handler.event_details = event_details
        print(f"Event Details: {update_handler.event_details}")
        print(f"Calendar Insights: {update_handler.calendar_insights}")

        if len(target_dates) > 1:
            update_handler.eliminate_targets(event_id)

        update_handler.update_event(event_id)

        return {"status": "success", "message": event_details.response}
    except Exception as e:
        print(f"I'm sorry, something went wrong. Please try again: {e}")
        return {"status": "failed", "message": f"Something went wrong please try again."}


@app.get("/")
def read_root():
    return {"Hello": "Jacey"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
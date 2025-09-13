from fastapi import FastAPI, HTTPException
from api.auth_router import router
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from src.model_setup.structure_model_output import EventDetails, HandleResponse
from src.google_calendar.handleDateTimes import DateTimeSet
from src.google_calendar.handleEvents import CalendarInsights, RequestSetup, AddToCalendar, DeleteFromCalendar, UpdateFromCalendar
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
)
app.include_router(router)

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


@app.post("/scheduler/process_input")
async def process_input(input_request: InputRequest) -> list[Dict[str, Any]]:
    """Processes user input, generates, a response, and returns that response as a list of dictionaries for either further processing or user feedback.

    Args:
        input_request (InputRequest): The user input request containing the input text.

    Raises:
        HTTPException: If an error occurs while processing the input request.

    Returns:
        list[Dict[str, Any]]: A list of dictionaries containing the response for each processed event. This is needed if the system needs to request additional information
    """
    try:
        input_text = input_request.input_text
        user_id = input_request.user_id
        
        response_handler = HandleResponse(input_text)
        events = response_handler.process_response() 
        requests = []
        
        for event in events:
            print(f"Processing Event: {event}")
            event_dict = {
                "event_name": event.event_name,
                "target_dates": event.datetime_obj.target_datetimes,
                "action": event.action,
                "response": event.response
            }

            if event.action.lower() == "add":
                add_handler = AddToCalendar(event, user_id)
                add_handler.add_event()
                result = add_handler.event_details.response

                responseRequest = process_response("completed", result, event_dict)
                requests.append(responseRequest)

            elif event.action.lower() == "delete":
                delete_handler = DeleteFromCalendar(event, user_id)
                delete_handler.find_matching_events()
                calendar_insights_dict = {
                    "matching_events": delete_handler.calendar_insights.matching_events,
                    "is_event": delete_handler.calendar_insights.is_event   
                }

                responseRequest = process_response("request event ID", f"Please select an event to delete as {event.event_name.upper()}", event_dict, calendar_insights_dict)
                requests.append(responseRequest)

            elif event.action.lower() == "update":
                update_handler = UpdateFromCalendar(event, user_id)
                update_handler.find_matching_events()
                calendar_insights_dict = {
                    "matching_events": update_handler.calendar_insights.matching_events,
                    "is_event": update_handler.calendar_insights.is_event
                }
                event_dict['target_dates'] = [(start, end) for start, end in update_handler.event_details.datetime_obj.target_datetimes]
                responseRequest = process_response("request event ID", f"Please select an event to update as {event.event_name.upper()}", event_dict, calendar_insights_dict)
                requests.append(responseRequest)

            else:
                result = f"Unknown action '{event.action}' for event '{event.event_name}'."
                responseRequest = process_response("failed", result, event_dict)    
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
    
@app.post("/scheduler/list_events")
async def list_events(event_request: EventRequest) -> list[dict]:
    event_details = event_request.event_details
    event_details.event_name = "None"
    event_details.action = "None"
    user_id = event_request.user_id
    request_setup = RequestSetup(event_details, user_id)
    events_list = request_setup.calendar_insights.scheduled_events
    events_json = []
    for event in events_list:
        event_dict = {
            "event_name": event.event_name,
            "start_time": event.start,
            "end_time": event.end,
            "event_id": event.event_id
        }
        print(f"Event JSON: {event_dict}")
        events_json.append(event_dict)
    return events_json

@app.post("/scheduler/update_event/{event_id}")
async def update_event(event_id: str, request_body: dict) -> dict:
    """Update an existing event in the calendar.

    Args:
        event_id (str): ID of the event to update.
        request_body (dict): The updated event details.

    Raises:
        HTTPException: If an error occurs while updating the event.

    Returns:
        dict: A dictionary containing the status and message of the update operation.
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
        calendar_insights_dict = request_body.get("calendar_insights", {})
        user_id = request_body.get("user_id", "None")
        calendar_insights = CalendarInsights(**calendar_insights_dict)
        
        update_handler = UpdateFromCalendar(event_details, user_id)
        update_handler.calendar_insights = calendar_insights
        update_handler.eliminate_targets(event_id)
        update_handler.fetch_event_template()
        update_handler.event_details = event_details

        print(f"Event Details: {event_details}")
        print(f"Calendar Insights: {calendar_insights}")
        update_handler.update_event(event_id, event_details, calendar_insights)

        return {"status": "success", "message": event_details.response}
    except Exception as e:
        print(f"I'm sorry, something went wrong. Please try again: {e}")
        return {"status": "failed", "message": f"Something went wrong please try again."}


@app.get("/")
def read_root():
    return {"Hello": "Jacey"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
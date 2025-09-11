from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.model_setup.structure_model_output import EventDetails, HandleResponse
from src.google_calendar.handleDateTimes import DateTimeSet
from src.google_calendar.handleEvents import CalendarInsights, AddToCalendar, DeleteFromCalendar, UpdateFromCalendar
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn


app = FastAPI()

# More restrictive CORS for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputRequest(BaseModel):
    input_text: str

responseRequest = {
    "status": str,
    "message": str,
    "event_requested": dict,
    "calendar_insights": Optional[dict]
}


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
        if not "." in input_text and not "?" in input_text:
            input_text += "."
        response_handler = HandleResponse(input_text)
        events = response_handler.process_response() 
        requests = []
        
        for event in events:
            print(f"Processing Event: {event.event_name}")
            event_dict = {
                "event_name": event.event_name,
                "target_dates": event.datetime_obj.target_datetimes,
                "action": event.action,
                "response": event.response
            }
            print(f"Event Dictionary: {event_dict}")
            print(f"Event datetime: {event.datetime_obj}")

            if event.action.lower() == "add":
                add_handler = AddToCalendar(event)
                add_handler.add_event()
                result = add_handler.event_details.response

                responseRequest = {
                    "status": "completed",
                    "message": result,
                    "event_requested": event_dict,
                    "calendar_insights": None
                }
                requests.append(responseRequest)

            elif event.action.lower() == "delete":
                delete_handler = DeleteFromCalendar(event)
                calendar_insights_dict = {
                    "matching_events": delete_handler.calendar_insights.matching_events,
                    "is_event": delete_handler.calendar_insights.is_event   
                }

                responseRequest = {
                    "status": "request event ID",
                    "message": f"Please select an event to delete as {event.event_name.upper()}",
                    "event_requested": event_dict,
                    "calendar_insights": calendar_insights_dict
                }
                requests.append(responseRequest)

            elif event.action.lower() == "update":
                update_handler = UpdateFromCalendar(event)
                calendar_insights_dict = {
                    "matching_events": update_handler.calendar_insights.matching_events,
                    "is_event": update_handler.calendar_insights.is_event
                }
                event_dict['target_dates'] = [(start, end) for start, end in update_handler.event_details.datetime_obj.target_datetimes]
                responseRequest = {
                    "status": "request event ID",
                    "message": f"Please select an event to update as {event.event_name.upper()}",
                    "event_requested": event_dict,
                    "calendar_insights": calendar_insights_dict
                }
                requests.append(responseRequest)

            else:
                result = f"Unknown action '{event.action}' for event '{event.event_name}'."
                responseRequest = {
                    "status": "failed",
                    "message": result,
                    "event_requested": event_dict,
                    "calendar_insights": None
                }
                requests.append(responseRequest)

        return requests
        
    except Exception as e:
        print(f"I'm sorry, something went wrong. Please try again: {e}")
        return [{"status": "failed", "message": f"Something went wrong please try again."}]

@app.post("/scheduler/delete_event/{event_id}")
async def delete_event(event_id: str, event_details: EventDetails) -> dict:
    """Delete an event from the calendar.

    Args:
        event_id (str): The ID of the event to delete.
        event_details (EventDetails): The details of the event which is needed to produce a response that was generated.

    Raises:
        HTTPException: If an error occurs while deleting the event.

    Returns:
        dict: A dictionary containing the status and message of the delete operation.
    """
    try:
        delete_handler = DeleteFromCalendar(event_details)
        delete_handler.delete_event(event_id)  
        return {"status": "success", "message": event_details.response}
    except Exception as e:
        print(f"I'm sorry, something went wrong. Please try again: {e}")
        return {"status": "failed", "message": f"Something went wrong please try again."}

@app.post("/scheduler/update_event/{event_id}")
async def update_event(event_id: str, request_body: dict):
    try:
        event_details_dict = request_body.get("event_details", {})
        calendar_insights_dict = request_body.get("calendar_insights", {})
        print(f"Received event_details: {event_details_dict}")

        datetime_set = DateTimeSet(
            target_datetimes=event_details_dict.get("target_dates", [])
        )
        event_details = EventDetails(
            event_name=event_details_dict.get("event_name", "None"),
            datetime_obj=datetime_set,
            action=event_details_dict.get("action", "None"),
            response=event_details_dict.get("response", "None")
        )
        calendar_insights = CalendarInsights(**calendar_insights_dict)
        print(f"Event Details: {event_details}")
        print(f"Calendar Insights: {calendar_insights}")

        update_handler = UpdateFromCalendar(event_details)
        update_handler.calendar_insights = calendar_insights
        update_handler.eliminate_targets(event_id)
        update_handler.event_details = event_details
        update_handler.update_event(event_id, event_details, calendar_insights)

        return {"status": "success", "message": event_details.response}
    except Exception as e:
        print(f"I'm sorry, something went wrong. Please try again: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        

@app.get("/")
def read_root():
    return {"Hello": "Jacey"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
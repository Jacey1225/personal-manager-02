from fastapi import HTTPException
import app
from google_calendar.handleEvents import HandleEvents
from structure_model_output import HandleResponse, EventDetails

@app.schedule_router.post("/process_input")
async def process_input(input_text: str):
    try:
        response_handler = HandleResponse(input_text)
        events = response_handler.process_response()
        for event in events:
            event_action = event.action.lower()
            if event_action in ["delete", "update"]:
                get_events = True
            else:
                get_events = False
            event_manager = HandleEvents(event, get_events=get_events)
            if event_action == "add":
                event_manager.create_event()
            if event_action == "delete":
                event_manager.delete_event()
            response_handler.convert_response_to_speech(event.response)
        return {"status": "success", "message": "Input processed successfully."}
    except Exception as e:
        app.logging.error(f"Error processing input: {e}")
        raise HTTPException(status_code=500, detail=str(e))
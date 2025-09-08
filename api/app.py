from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import sys
import os

# Add the src directory to the path so imports work
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from google_calendar.handleEvents import HandleEvents
from structure_model_output import HandleResponse

app = FastAPI()

class InputRequest(BaseModel):
    input_text: str

@app.post("/scheduler/process_input")
async def process_input(request: InputRequest):
    try:
        input_text = request.input_text
        if not "." in input_text and not "?" in input_text:
            input_text += "."
            
        response_handler = HandleResponse(input_text)
        events = response_handler.process_response()
        
        for event in events:
            calendar_handler = HandleEvents(event)
            print(f"Processing Event: {calendar_handler.event_details}")
            
            if event.action.lower() == "add":
                result = calendar_handler.add_event()
            elif event.action.lower() == "delete":
                result = calendar_handler.delete_event()
            elif event.action.lower() == "update":
                result = calendar_handler.update_event()
            else:
                result = f"Unknown action '{event.action}' for event '{event.event_name}'."
            
            return {
                "status": "success", 
                "message": calendar_handler.event_details.response
            }
            
    except Exception as e:
        print(f"Error processing input: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"Hello": "Jacey"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
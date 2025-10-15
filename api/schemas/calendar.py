from pydantic import BaseModel, Field
from api.schemas.model import EventDetails
from typing import Optional
from datetime import datetime, time


class CalendarEvent(BaseModel): 
    """used to gather information on events found within the calendar

    Raises:
        TypeError: Start time conflict
        TypeError: End time conflict
    """
    event_name: str = Field(default="None", description="The title of the event or task")
    start: datetime = Field(default=datetime.now(), description="The start datetime of the event")
    end: Optional[datetime] = Field(default=None, description="The end datetime of the event only if classified as an event")
    description: str = Field(default="None", description="A brief description of the event")
    timezone: str = Field(default="America/Los_Angeles", description="The timezone of the event")
    is_event: bool = Field(default=False, description="Determines whether we want to handle the request as an event or task")
    event_id: str = Field(default="None", description="The unique identifier for the event")

class CalendarInsights(BaseModel): 
    """used to store additional information on the requested event regarding what already exists in the calendar 

    Raises:
        TypeError: Start time conflict
        TypeError: End time conflict
        TypeError: Event ID conflict
        TypeError: Template conflict
    """
    services: dict = Field(default={}, description="A list of all services used in the calendar(Google, Apple, etc.)")
    scheduled_events: list[CalendarEvent] = Field(default=[], description="List of all processed events existing in the calendar")
    matching_events: list[dict] = Field(default=[], description="List of all matching events found in the calendar")
    project_events: list[dict] = Field(default=[], description="List of all project events found in the calendar")
    template: dict = Field(default={}, description="A template used to send the calendar API event info")
    is_event: bool = Field(default=False, description="Determines whether we want to handle the request as an event or task")
    selected_event_id: Optional[str] = Field(default="None", description="The event ID of the selected event to be updated or deleted")
    
class EventRequest(BaseModel):
    event_details: EventDetails
    target_start: str | None = None
    target_end: str | None = None
    user_id: str

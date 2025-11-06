from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, time

class DateTimeSet(BaseModel):
    input_tokens: list[str] = Field(default=[], description="A list of all the input tokens found within an input text")
    times: list[time] = Field(default=[], description="A list of all the times found within an input text")
    dates: list[datetime] = Field(default=[], description="A list of all the dates found within an input text")
    datetimes: list[datetime] = Field(default=[], description="A list of all the datetime objects found within an input text")
    target_datetimes: list[tuple] = Field(default=[], description="A list of all the target datetime objects found within an input text as tuples representing start and end or due and None")

class CalendarEvent(BaseModel): 
    """used to gather information on events found within the calendar

    Raises:
        TypeError: Start time conflict
        TypeError: End time conflict
        TypeError: Event ID conflict
        TypeError: Template conflict
    """
    event_name: str = Field(default="None", description="The title of the event or task")
    datetime_obj: DateTimeSet = Field(default_factory=DateTimeSet, description="List of datetime objects extracted from the input text")
    start: datetime = Field(default=datetime.now(), description="The start datetime of the event")
    end: datetime = Field(default=datetime.now(), description="The end datetime of the event")
    description: str = Field(default="", description="A brief description of the event")
    status: Optional[str] = Field(default="pending", description="The status of the event (e.g., pending, complete)")
    transparency: str = Field(default="transparent", description="The transparency of the event (opaque, transparent, etc.)")
    guestsCanModify: bool = Field(default=False, description="Indicates if guests can modify the event")
    attendees: List[str] = Field(default_factory=list, description="List of attendees for the event")
    timezone: Optional[str] = Field(default="America/Los_Angeles", description="The timezone of the event")
    is_event: Optional[bool] = Field(default=False, description="Determines whether we want to handle the request as an event or task")
    event_id: Optional[str] = Field(default=None, description="The unique identifier for the event")
    calendar_id: Optional[str] = Field(default=None, description="The unique identifier for the calendar")

    def to_apple(self):
        return {
            "pguid": self.calendar_id,
            "guid": self.event_id,
            "title": self.event_name,
            "startDate": self.start,
            "endDate": self.end,
            "status": self.status,
            "description": self.description
        }
    def to_google(self):
        return {
            "id": self.event_id,
            "summary": self.event_name,
            "start": {
                "dateTime": self.start,
                "timeZone": self.timezone
            },
            "end": {
                "dateTime": self.end,
                "timeZone": self.timezone
            },
            "status": self.status,
            "description": self.description
        }
class CalendarInsights(BaseModel): 
    """used to store additional information on the requested event regarding what already exists in the calendar 

    Raises:
        TypeError: Start time conflict
        TypeError: End time conflict
        TypeError: Event ID conflict
        TypeError: Template conflict
    """
    scheduled_events: list[Optional[CalendarEvent]] = Field(default=[], description="List of all processed events existing in the calendar")
    matching_events: list[CalendarEvent] = Field(default=[], description="List of all matching events found in the calendar related to the user's request")
    project_events: list[dict] = Field(default=[], description="List of all project events found in the calendar")
    template: Optional[dict] = Field(default={}, description="A template used to send the calendar API event info")
    
class EventsRequest(BaseModel):
    calendar_event: CalendarEvent
    minTime: str | None = None 
    maxTime: str | None = None
    user_id: str

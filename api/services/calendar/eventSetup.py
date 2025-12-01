from typing import Optional
from datetime import datetime, timedelta
from api.config.plugins.enable_google_api import SyncGoogleEvents, SyncGoogleTasks
from api.config.plugins.enable_apple_api import SyncAppleEvents
from api.config.uniformInterface import UniformInterface
from api.config.fetchMongo import MongoHandler
from api.services.calendar.handleDateTimes import DateTimeHandler
from api.schemas.calendar import CalendarEvent, CalendarInsights   
from api.schemas.model import EventOutput
from api.validation.handleEventSetup import ValidateEventHandling
import pytz

validator = ValidateEventHandling()
user_config = MongoHandler("userAuthDatabase", "userCredentials")
class RequestSetup:
    """Handles event setup for Google Calendar and Google Tasks.
    """
    def __init__(self, 
                 calendar_event: CalendarEvent, 
                 event_output: EventOutput,
                 user_id: str, 
                 calendar_service, 
                 minTime: Optional[str] = (datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).isoformat() + 'Z', 
                 maxTime: Optional[str] = (datetime.now().replace(hour=23, minute=59, second=59, microsecond=0) + timedelta(days=30)).isoformat() + 'Z',
                 description: Optional[str] = None):
        self.user_id = user_id
        self.minTime = minTime
        self.maxTime = maxTime
        self.description = description
        self.event_output = event_output
        self.calendar_event = calendar_event
        self.calendar_service: Optional[SyncAppleEvents | SyncGoogleEvents | SyncGoogleTasks] = calendar_service
        self.datetime_handler = DateTimeHandler(self.event_output.input_text)
        self.calendar_insights = CalendarInsights()

        # Only validate calendar event data if calendar service is provided
        # This allows non-calendar operations to work without calendar events
        if self.calendar_service is not None and (not self.calendar_event.event_name or not self.event_output.intent):
            print(f"Event Name: {self.calendar_event.event_name}")
            print(f"Event Action: {self.event_output.intent}")
            print(f"Input Text: {self.event_output.input_text}")
            raise ValueError("Event name and action must be provided in event_output.")
        
        # For non-calendar operations, initialize default calendar insights
        if self.calendar_service is None:
            self.calendar_insights = CalendarInsights()

    @validator.validate_events_list
    def fetch_events_list(self):
        """Fetch the list of events from Google Calendar and Google Tasks.

        Raises:
            ConnectionError: If unable to connect to Google Calendar or Google Tasks API.
            RuntimeError: If an error occurs while fetching events.
        """
        try:
            if self.calendar_service is None:
                raise ConnectionError("Calendar service is not initialized.")
            scheduled_events = self.calendar_service.list_events(
                maxResults=100,
                timeMin=self.calendar_event.start,
                timeMax=self.calendar_event.end,
                description=self.calendar_event.description
            )
            self.calendar_insights.scheduled_events = scheduled_events
        except Exception as e:
            print(f"Error fetching tasks: {e}")

    def fetch_event_template(self):
        """Fetch the event template for creating a new event.

        Raises:
            ValueError: If the event template cannot be fetched.
        """
        try:
            if self.calendar_service is None:
                raise ConnectionError("Calendar service is not initialized.")
            self.calendar_insights.template = self.calendar_service.template(
                **self.calendar_event.__dict__)
        except:
            print(f"Error fetching event template in: {self.fetch_event_template.__name__} with Calendar Event: {self.calendar_event.__dict__}")
            self.calendar_insights.template = None

        if not self.calendar_insights.template:
            raise ValueError("Failed to fetch event template.")
        return self.calendar_insights.template

    @validator.log_matching_events
    def find_matching_events(self) -> list[CalendarEvent]:
        """Fetch the events currently in the calendar that closely match the event of interest 

        Returns:
            list[CalendarEvent]: A list of matching CalendarEvent objects.
        """
        for token in self.calendar_event.event_name.lower().split(" "):
            for event in self.calendar_insights.scheduled_events:
                if event and token in event.event_name.lower():
                    if event not in self.calendar_insights.matching_events:
                        self.calendar_insights.matching_events.append(event)
                    else:
                        continue
        return self.calendar_insights.matching_events

    def tie_project(self, 
                    user_data: dict) -> CalendarEvent:  # Not part of the model API
        """Ties a project to the event details based on the project name.

        Args:
            event_details (EventDetails): The details of the event being processed.

        Returns:
            EventDetails: The updated event details with project information.
        """
        try:
            for key, (project_name, permission) in user_data["projects"].items():
                if permission == "view":
                    continue
                if project_name.lower() in self.event_output.input_text.lower():
                    self.calendar_event.transparency = "transparent"
                    self.calendar_event.guestsCanModify = True
                    self.calendar_event.description = f"Lazi: {key}"
                    break

            return self.calendar_event
        except Exception as e:
            print(f"Error tying project to event details: {e}")
            print(f"Function: {self.tie_project.__name__}")
            return self.calendar_event
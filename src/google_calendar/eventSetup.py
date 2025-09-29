from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timedelta
from src.google_calendar.enable_google_api import ConfigureGoogleAPI
from src.model_setup.structure_model_output import EventDetails
from src.google_calendar.handleDateTimes import DateTimeHandler
from src.validators.handleEventSetup import ValidateEventHandling

validator = ValidateEventHandling()
def check_service(event_service, task_service):
    if event_service is None:
        raise ConnectionError("Failed to connect to Google Calendar API.")
    if task_service is None:
        raise ConnectionError("Failed to connect to Google Tasks API.")
    return True

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
    scheduled_events: list[CalendarEvent] = Field(default=[], description="List of all processed events existing in the calendar")
    matching_events: list[dict] = Field(default=[], description="List of all matching events found in the calendar")
    project_events: list[dict] = Field(default=[], description="List of all project events found in the calendar")
    template: dict = Field(default={}, description="A template used to send the calendar API event info")
    is_event: bool = Field(default=False, description="Determines whether we want to handle the request as an event or task")
    selected_event_id: Optional[str] = Field(default="None", description="The event ID of the selected event to be updated or deleted")

class RequestSetup:
    """Handles event setup for Google Calendar and Google Tasks.
    """
    def __init__(self, event_details: EventDetails, user_id: str, personal_email: str = "jaceysimps@gmail.com"):
        self.user_id = user_id
        self.multi_user_google_api = ConfigureGoogleAPI(user_id)
        try:
            result = self.multi_user_google_api.enable_google_calendar_api()
            
            self.event_service = None
            self.task_service = None
            if isinstance(result, str):
                raise ConnectionError(f"User authentication required. Please complete Google OAuth first. Auth URL: {result}")
            elif result is not None and len(result) == 2:
                self.event_service, self.task_service = result
            else:
                raise ConnectionError("Failed to initialize Google API services")
        except Exception as e:
            print(f"Error initializing Google API services: {e}")
        self.personal_email = personal_email
        self.event_details = event_details
        self.datetime_handler = DateTimeHandler(self.event_details.input_text)
        self.task_list_id: str = '@default'
        self.event_list_id: str = 'primary'

        if not self.event_details.event_name or not self.event_details.action:
            print(f"Event Name: {self.event_details.event_name}")
            print(f"Event Action: {self.event_details.action}")
            print(f"Input Text: {self.event_details.input_text}")
            raise ValueError("Event name and action must be provided in event_details.")
        
#MARK: Request Setup
        self.calendar_insights = CalendarInsights(
            scheduled_events=[],
            template={},
            is_event=False
        )
        self.fetch_events_list()
        self.calendar_insights.scheduled_events = self.datetime_handler.sort_datetimes(self.calendar_insights.scheduled_events)

    def validate_event_obj(self, event_obj: CalendarEvent) -> tuple[datetime, Optional[datetime]]:
        """Validates and formats the start and end times of a calendar event.

        Args:
            event_obj (CalendarEvent): The event object to validate.

        Returns:
            tuple[datetime, Optional[datetime]]: The validated start and end times.
        """
        if isinstance(event_obj.start, str):
            event_obj.start = datetime.fromisoformat(event_obj.start)
            if event_obj.start.tzinfo is not None:
                event_obj.start = event_obj.start.astimezone()
                event_obj.start = event_obj.start.replace(tzinfo=None)
                
        if isinstance(event_obj.end, str):
            event_obj.end = datetime.fromisoformat(event_obj.end)
            if event_obj.end.tzinfo is not None:
                event_obj.end = event_obj.end.astimezone()
                event_obj.end = event_obj.end.replace(tzinfo=None)
                
        return event_obj.start, event_obj.end

    @validator.validate_events_list
    def fetch_events_list(self):
        """Fetch the list of events from Google Calendar and Google Tasks.

        Raises:
            ConnectionError: If unable to connect to Google Calendar or Google Tasks API.
            RuntimeError: If an error occurs while fetching events.
        """
        try:
            if not check_service(self.event_service, self.task_service):
                raise ConnectionError("Google Calendar or Tasks service is not available.")
        
            try:
                tasks_list = self.task_service.tasks().list( #type:ignore
                    tasklist=self.task_list_id,
                    dueMin=datetime.now().isoformat() + 'Z',
                ).execute() 
            except Exception as e:
                print(f"Error fetching tasks: {e}")
                tasks_list = {'items': []}
            
            try:
                events_list = self.event_service.events().list( #type:ignore
                    calendarId=self.event_list_id,
                    timeMin=datetime.now().isoformat() + 'Z',
                    timeMax=(datetime.now() + timedelta(days=30)).isoformat() + 'Z',
                ).execute() 
            except Exception as e:
                print(f"Error fetching events: {e}")
                print(f"Function: {self.fetch_events_list.__name__}")
                events_list = {'items': []}
        
            scheduled_events = events_list.get('items', []) + tasks_list.get('items', [])
            for event in scheduled_events:
                event_obj = CalendarEvent(
                    event_name='',
                    start=datetime.now(),
                    end=None,
                    is_event=False
                )
                if 'summary' in event:
                    event_obj.event_name = event['summary']
                    event_obj.is_event = True
                    event_obj.start = (event.get('start', {}).get('dateTime'))
                    event_obj.end = event.get('end', {}).get('dateTime')
                    event_obj.description = event.get('description', 'No description provided')
                    event_obj.event_id = event.get('id')
                elif 'title' in event:
                    event_obj.event_name = event['title']
                    event_obj.start = event.get('due')
                    event_obj.description = event.get('description', 'No description provided')
                    event_obj.event_id = event.get('id')

                if event_obj.start:
                    event_obj.start, event_obj.end = self.validate_event_obj(event_obj)
                else:
                    Warning(f"No valid start or end time found for event: {event}")
                    continue

                self.calendar_insights.scheduled_events.append(event_obj)
        except AttributeError as e:
            print(f"Service attribute error: {e}")
            pass
        except Exception as e:
            raise RuntimeError(f"An error occurred while fetching events: {e}")

    def fetch_event_template(self):
        """Fetch the event template for creating a new event.

        Raises:
            ValueError: If event name or personal email is not provided.
        """
        if not self.calendar_insights.is_event:
            self.calendar_insights.template = {
                'title': self.event_details.event_name,
                'due': None,  # This will hold the RFC 3339 timestamp for due date and time
            }
        else:
            self.calendar_insights.template = {
                'summary': self.event_details.event_name,
                'description': self.event_details.description,
                'transparency': self.event_details.transparency,
                'guestsCanModify': self.event_details.guestsCanModify,
                'start': {
                    'dateTime': None,
                    'timeZone': 'America/Los_Angeles',
                },
                'end': {
                    'dateTime': None,
                    'timeZone': 'America/Los_Angeles',
                },
                'attendees': [{'email': self.personal_email}],
            }

    @validator.log_matching_events
    def find_matching_events(self) -> list[dict]:
        """Fetch the events currently in the calendar that closely match the event of interest 

        Returns:
            list[dict]: A list of matching event dictionaries.
        """
        for token in self.event_details.event_name.lower().split(" "):
            for event in self.calendar_insights.scheduled_events:
                if token in event.event_name.lower():
                    event_dict = {
                        "event_name": event.event_name,
                        "start": event.start.isoformat(),
                        "end": event.end.isoformat() if event.end else None,
                        "is_event": event.is_event,
                        "event_id": event.event_id
                    }
                    if event_dict not in self.calendar_insights.matching_events:
                        self.calendar_insights.matching_events.append(event_dict)
                    else:
                        continue
        return self.calendar_insights.matching_events
    
    @validator.validate_request_classifier
    def classify_request(self, target_datetime: Optional[tuple]=None, event_id: Optional[str]=None):
        """Classify the request based on the target datetime and event ID.

        Args:
            target_datetime (tuple): The target datetime for the event.
            event_id (Optional[str], optional): The ID of the event. Defaults to None.
        """
        if event_id and self.calendar_insights.matching_events:
            calendar_target = next((event for event in self.calendar_insights.matching_events if event['event_id'] == event_id), None)
            if calendar_target:
                if calendar_target['is_event']:
                    self.calendar_insights.is_event = True
                else:
                    self.calendar_insights.is_event = False
        else:
            if target_datetime:
                if target_datetime[1]:
                    self.calendar_insights.is_event = True
                else:
                    self.calendar_insights.is_event = False
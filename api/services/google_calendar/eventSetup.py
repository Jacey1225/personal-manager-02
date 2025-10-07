from typing import Optional
from datetime import datetime, timedelta
from api.services.google_calendar.enable_google_api import ConfigureGoogleAPI
from api.services.model_setup.structure_model_output import EventDetails
from api.services.google_calendar.handleDateTimes import DateTimeHandler
from api.schemas.calendar import CalendarEvent, CalendarInsights   
from api.validation.handleEventSetup import ValidateEventHandling
import pytz

validator = ValidateEventHandling()
def check_service(event_service, task_service):
    if event_service is None:
        raise ConnectionError("Failed to connect to Google Calendar API.")
    if task_service is None:
        raise ConnectionError("Failed to connect to Google Tasks API.")
    return True

class RequestSetup:
    """Handles event setup for Google Calendar and Google Tasks.
    """
    def __init__(self, event_details: EventDetails, user_id: str, personal_email: str = "jaceysimps@gmail.com", 
                 minTime: str | None = (datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).isoformat() + 'Z', 
                 maxTime: str | None = (datetime.now().replace(hour=23, minute=59, second=59, microsecond=0) + timedelta(days=30)).isoformat() + 'Z',
                 description: str | None = None):
        self.user_id = user_id
        self.multi_user_google_api = ConfigureGoogleAPI(user_id)
        self.minTime = minTime
        self.maxTime = maxTime
        self.description = description
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
    def validate_event_obj(self, event_obj: CalendarEvent) -> tuple[datetime, Optional[datetime]]:
        """Validates and formats the start and end times of a calendar event.

        Args:
            event_obj (CalendarEvent): The event object to validate.

        Returns:
            tuple[datetime, Optional[datetime]]: The validated start and end times.
        """
        event_tz = None
        if event_obj.timezone:
            try:
                event_tz = pytz.timezone(event_obj.timezone)
            except Exception as e:
                event_tz = pytz.timezone("America/Los_Angeles")

        if isinstance(event_obj.start, str):
            event_obj.start = datetime.fromisoformat(event_obj.start)
            if event_obj.start.tzinfo is not None:
                event_obj.start = event_obj.start.astimezone(event_tz)                
        if isinstance(event_obj.end, str):
            event_obj.end = datetime.fromisoformat(event_obj.end)
            if event_obj.end.tzinfo is not None:
                event_obj.end = event_obj.end.astimezone(event_tz)

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
                if self.description is None:
                    tasks_list = self.task_service.tasks().list( #type:ignore
                        tasklist=self.task_list_id,
                        maxResults="50",
                        dueMin=self.minTime,
                        dueMax=self.maxTime
                    ).execute() 
                else:
                    tasks_list = {'items': []}
            except Exception as e:
                print(f"Error fetching tasks: {e}")
                tasks_list = {'items': []}
            
            try:
                print(f"Fetching events with description filter: {self.description}")
                events_list = self.event_service.events().list( #type:ignore
                    calendarId=self.event_list_id,
                    maxResults=100,
                    q=self.description if self.description else "",
                    timeMin=self.minTime,
                    timeMax=self.maxTime,
                    singleEvents=True,
                    orderBy='startTime'
                ).execute() 
            except Exception as e:
                print(f"Error fetching events in {self.fetch_events_list.__name__}: {e}")
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
                    event_obj.timezone = event.get('start', {}).get('timeZone', "America/Los_Angeles")
                    event_obj.event_id = event.get('id')
                elif 'title' in event:
                    event_obj.event_name = event['title']
                    event_obj.start = event.get('due')
                    event_obj.description = event.get('description', 'No description provided')
                    event_obj.timezone = event.get('start', {}).get('timeZone', "America/Los_Angeles")
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
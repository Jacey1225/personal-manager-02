from src.model_setup.structure_model_output import EventDetails
from src.google_calendar.handleDateTimes import DateTimeHandler
from src.google_calendar.enable_google_api import ConfigureGoogleAPI
from src.validators.validators import ValidateEventHandling
from datetime import datetime, timezone
import pytz
from pydantic import BaseModel, Field
from typing import Optional, Union

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
    is_project: Optional[bool] = Field(default=False, description="Indicates that the event or task is contained within a project")
    start: datetime = Field(default=datetime.now(), description="The start datetime of the event")
    end: Optional[datetime] = Field(default=None, description="The end datetime of the event only if classified as an event")
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
            
            if isinstance(result, str):
                # Auth URL returned - user needs to authenticate
                raise ConnectionError(f"User authentication required. Please complete Google OAuth first. Auth URL: {result}")
            elif result is not None and len(result) == 2:
                # Services returned - user is authenticated
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
        self.find_matching_events()

    def validate_event_obj(self, event_obj: CalendarEvent) -> tuple[datetime, Optional[datetime]]:
        if isinstance(event_obj.start, str):
            event_obj.start = datetime.fromisoformat(event_obj.start)
        if isinstance(event_obj.end, str):
            event_obj.end = datetime.fromisoformat(event_obj.end)
        return event_obj.start, event_obj.end

    @validator.validate_events_list
    def fetch_events_list(self):
        """Fetch the list of events from Google Calendar and Google Tasks.

        Raises:
            ConnectionError: If unable to connect to Google Calendar or Google Tasks API.
            RuntimeError: If an error occurs while fetching events.
        """
        try:
            if not check_service(self.event_service, self.task_service): #fetch existing events if needed
                raise ConnectionError("Google Calendar or Tasks service is not available.")

            tasks_list = self.task_service.tasks().list(tasklist=self.task_list_id).execute() #type: ignore
            events_list = self.event_service.events().list(calendarId=self.event_list_id).execute() #type: ignore
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
                    event_obj.event_id = event.get('id')
                elif 'title' in event:
                    event_obj.event_name = event['title']
                    event_obj.start = event.get('due')
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
                'description': '',
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
                    if self.datetime_handler.verify_event_time(event.start):
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

#MARK: add
class AddToCalendar(RequestSetup):
    def __init__(self, event_details: EventDetails, user_id: str, personal_email: str = "jaceysimps@gmail.com"):
        super().__init__(event_details, user_id, personal_email)

    @validator.validate_request_status
    def add_event(self):
        """Add a new event to Google Calendar or a task to Google Tasks.

        Raises:
            ValueError: If event template is not initialized.
            ValueError: If no datetime objects are provided.
            RuntimeError: If an error occurs while adding the event/task.

        Returns:
            str: A message indicating the result of the operation.
        """
        try:
                for start_time, end_time in self.event_details.datetime_obj.target_datetimes:
                    self.classify_request((start_time, end_time))
                    self.fetch_event_template()
                    if self.calendar_insights.is_event:
                        self.calendar_insights.template['start']['dateTime'] = start_time.isoformat()
                        self.calendar_insights.template['end']['dateTime'] = end_time.isoformat()
                        if start_time and end_time:
                            self.event_service.events().insert(calendarId=self.event_list_id, body=self.calendar_insights.template).execute() #type: ignore
                    else:
                        local_tz = pytz.timezone('America/Los_Angeles')
                        due_datetime = local_tz.localize(start_time)
                        self.calendar_insights.template['due'] = due_datetime.isoformat()
                        if due_datetime:
                            self.task_service.tasks().insert(tasklist=self.task_list_id, body=self.calendar_insights.template).execute() #type: ignore

                return {"status": "success"}
        except Exception as e:
            print(f"An error occurred while adding the event/task: {e}")
            return {"status": "error", "message": str(e)}

#MARK: delete
class DeleteFromCalendar(RequestSetup):
    def __init__(self, event_details: EventDetails, user_id: str, personal_email: str = "jaceysimps@gmail.com"):
        super().__init__(event_details, user_id, personal_email)

    @validator.validate_request_status
    def delete_event(self, event_id: str):
        """Delete an event from Google Calendar or a task from Google Tasks.

        Raises:
            ValueError: If no event is found to delete.
            RuntimeError: If an error occurs while deleting the event/task.

        Returns:
            str: A message indicating the result of the operation.
        """
        try:
            if event_id:
                calendar_event = next((event for event in self.calendar_insights.matching_events if event['event_id'] == event_id), None)
                if calendar_event:
                    if calendar_event['is_event']:
                        self.calendar_insights.is_event = True
                if not calendar_event:
                    raise ValueError(f"No event found with ID '{event_id}'.")

                self.classify_request(event_id=event_id)
                if self.calendar_insights.is_event:
                    self.event_service.events().delete(calendarId=self.event_list_id, eventId=event_id).execute() #type: ignore
                else:
                    self.task_service.tasks().delete(tasklist=self.task_list_id, task=event_id).execute() #type: ignore
                return {"status": "success"}
            else:
                raise ValueError(f"No event found with ID '{event_id}'.")

        except Exception as e:
            raise RuntimeError(f"An error occurred while deleting the event/task: {e}")
        
#MARK: Update
class UpdateFromCalendar(RequestSetup):
    def __init__(self, event_details: EventDetails, user_id: str, personal_email: str = "jaceysimps@gmail.com"):
        super().__init__(event_details, user_id, personal_email)

    @validator.log_target_elimination
    def eliminate_targets(self, event_id: str):
        """Eliminate target datetimes for a specific event ID to avoid irrelevant datetime instances

        Args:
            event_id (str): The ID of the event to eliminate targets for.
        """
        calendar_target = next((event for event in self.calendar_insights.matching_events if event['event_id'] == event_id), None)
        if calendar_target:
            self.classify_request(event_id=event_id)

        all_target_datetimes = self.event_details.datetime_obj.target_datetimes.copy()
        for target_datetime in all_target_datetimes:
            start_target, end_target = target_datetime
            start_target = datetime.fromisoformat(start_target).replace(tzinfo=timezone.utc).astimezone(tz=None).isoformat()
            end_target = datetime.fromisoformat(end_target).replace(tzinfo=timezone.utc).astimezone(tz=None).isoformat() if end_target else None
            if calendar_target:
                calendar_start = datetime.fromisoformat(calendar_target['start']).replace(tzinfo=timezone.utc).astimezone(tz=None).isoformat()
                calendar_end = datetime.fromisoformat(calendar_target['end']).replace(tzinfo=timezone.utc).astimezone(tz=None).isoformat() if calendar_target['end'] else None
                if calendar_start and calendar_end:
                    if start_target == calendar_start and end_target == calendar_end:
                        self.event_details.datetime_obj.target_datetimes.remove(target_datetime)
                if calendar_start and not calendar_end:
                    if calendar_start == start_target:
                        self.event_details.datetime_obj.target_datetimes.remove(target_datetime)

    @validator.validate_request_status
    def update_event(self, event_id: str, event_details: EventDetails, calendar_insights: CalendarInsights):
        """Update an existing event in Google Calendar or a task in Google Tasks.

        Raises:
            ValueError: If event name or personal email is not provided.
            RuntimeError: If an error occurs while updating the event/task.

        Returns:
            str: A message indicating the result of the operation.
        """
        if not event_details.datetime_obj.target_datetimes:
            raise ValueError("At least one datetime object must be provided to update an event.")
        
        try:
            if event_id:
                if calendar_insights.is_event:
                    for start_time, end_time in event_details.datetime_obj.target_datetimes:
                        original_event = self.event_service.events().get(calendarId=self.event_list_id, eventId=event_id).execute() #type: ignore
                        original_event['start']['dateTime'] = start_time
                        original_event['end']['dateTime'] = end_time
                        self.event_service.events().update(calendarId=self.event_list_id, eventId=event_id, body=original_event).execute() #type: ignore
                else:
                    for due_time, _ in event_details.datetime_obj.target_datetimes:
                        original_task = self.task_service.tasks().get(tasklist=self.task_list_id, task=event_id).execute() #type: ignore
                        local_tz = pytz.timezone('America/Los_Angeles')
                        due_datetime = local_tz.localize(due_time)
                        original_task['due'] = due_datetime.isoformat()  # Use 'due' field for task due date/time
                    self.task_service.tasks().update(tasklist=self.task_list_id, task=event_id, body=original_task).execute() #type: ignore
                return {"status": "success"}
            else:
                raise ValueError(f"No event found with ID '{event_id}'.")

        except Exception as e:
            raise RuntimeError(f"An error occurred while updating the event/task: {e}")
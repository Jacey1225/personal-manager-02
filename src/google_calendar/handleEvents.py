from src.structure_model_output import EventDetails, HandleResponse
from src.google_calendar.enable_google_api import enable_google_calendar_api
from datetime import datetime, timezone
import pytz
from pydantic import BaseModel, Field
from typing import Optional, Union

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

    def validate_event_vulnerabilities(self):
        if not self.start or not isinstance(self.start, datetime):
            raise TypeError("Start time must be a datetime object.")
        if not isinstance(self.end, Union[datetime, type(None)]):
            raise TypeError("End time must be a datetime object if provided.")

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

    def validate_insight_vulnerabilities(self):
        for event in self.scheduled_events:
            if not isinstance(event, CalendarEvent):
                raise TypeError("Scheduled events must be instances of CalendarEvent.")
        if self.is_event:
            start_value = self.template['start'] 
            end_value = self.template['end']
            if not start_value or not isinstance(start_value["dateTime"], str) or "T" not in start_value["dateTime"]:
                raise TypeError(f"Template start dateTime must be a string in ISO format: {start_value['dateTime']}")
            if not end_value or not isinstance(end_value["dateTime"], str) or "T" not in end_value["dateTime"]:
                raise TypeError(f"Template end dateTime must be a string in ISO format: {end_value['dateTime']}")
        else:
            due_value = self.template['due']
            if not due_value or not isinstance(due_value, str) or "T" not in due_value:
                raise TypeError(f"Template due dateTime must be a string in ISO format: {due_value}")

class HandleEvents:
    """Handles event setup for Google Calendar and Google Tasks.
    """
    def __init__(self, event_details: EventDetails, personal_email: str = "jaceysimps@gmail.com"):
        self.event_service, self.task_service = enable_google_calendar_api() 
        self.personal_email = personal_email
        self.event_details = event_details
        self.task_list_id: str = '@default'
        self.event_id: str = 'primary'

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
        if not self.event_details.action == "delete":
            self.fetch_targets()
        self.fetch_event_template()
        
    def verify_event_time(self, event_start: Union[str, datetime]):
        current_date = datetime.now()
        if isinstance(event_start, str):
            event_start = datetime.fromisoformat(event_start)

        if event_start.strftime('%Y-%m-%d') < current_date.strftime('%Y-%m-%d'):
            return False
        return True

    def validate_event_obj(self, event_obj: CalendarEvent):
        if isinstance(event_obj.start, str):
            event_obj.start = datetime.fromisoformat(event_obj.start)
        if isinstance(event_obj.end, str):
            event_obj.end = datetime.fromisoformat(event_obj.end)
        return event_obj.start, event_obj.end

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
            events_list = self.event_service.events().list(calendarId=self.event_id).execute() #type: ignore
            scheduled_events = events_list.get('items', []) + tasks_list.get('items', []) #type: ignore
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
                    event_obj.start = event.get('updated')
                    event_obj.event_id = event.get('id')

                if event_obj.start:
                    event_obj.start, event_obj.end = self.validate_event_obj(event_obj)
                else:
                    Warning(f"No valid start or end time found for event: {event}")
                    continue

                self.calendar_insights.scheduled_events.append(event_obj)
                self.calendar_insights.scheduled_events[-1].validate_event_vulnerabilities()
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
        if not self.event_details.event_name or not self.personal_email:
            raise ValueError("Event name and personal email must be provided.")
        if not self.calendar_insights.is_event:
            self.calendar_insights.template = {
                'title': self.event_details.event_name,
                'due': None,
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

    def fetch_targets(self, zone:str = '-07:00'): #FIX: Dont fetch for datetime lengths of 2
        """Fetch the target dates for the event or task and set self.calendar_insights.is_event.
        """
        event_start = None
        event_end = None
        deleted_events = []
        if not self.event_details.datetime_obj.datetimes:
            print(f"No valid datetimes found for event: {self.event_details.event_name}")
            return
        
        
        for existing_event in self.calendar_insights.matching_events:
            datetime_objs = self.event_details.datetime_obj.datetimes.copy()
            for dt in datetime_objs:
                target_time = dt.isoformat() + zone
                if existing_event['start'].isoformat() == target_time and dt not in deleted_events:
                    self.event_details.datetime_obj.datetimes.remove(dt)
                    deleted_events.append(dt)
                    continue
                if existing_event['end'] and existing_event['end'].isoformat() == target_time and dt not in deleted_events:
                    self.event_details.datetime_obj.datetimes.remove(dt)
                    deleted_events.append(dt)
                    continue

        if len(self.event_details.datetime_obj.datetimes) < 2:

            self.calendar_insights.is_event = False
        else:
            self.calendar_insights.is_event = True

        if self.calendar_insights.is_event:
            event_start = min(self.event_details.datetime_obj.datetimes)
            event_end = max(self.event_details.datetime_obj.datetimes)
            print(f"Setting event start to {event_start.isoformat()} and end to {event_end.isoformat()}")
            self.event_details.datetime_obj.target_datetimes.append((event_start.isoformat(), event_end.isoformat()))
        else:
            event_start = min(self.event_details.datetime_obj.datetimes)
            print(f"Setting task due to {event_start.isoformat()}")
            self.event_details.datetime_obj.target_datetimes.append((event_start.isoformat(), None))

    def find_matching_events(self) -> list[dict]:
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
                    self.calendar_insights.matching_events.append(event_dict)

        return self.calendar_insights.matching_events

#MARK: add

class AddToCalendar(HandleEvents):
    def __init__(self, event_details: EventDetails, personal_email: str = "jaceysimps@gmail.com"):
        super().__init__(event_details, personal_email)

    def add_event(self):
        """Add a new event to Google Calendar or a task to Google Tasks.

        Raises:
            ValueError: If event template is not initialized.
            ValueError: If no datetime objects are provided.
            RuntimeError: If an error occurs while adding the event/task.

        Returns:
            str: A message indicating the result of the operation.
        """
        if not self.calendar_insights.template:
            raise ValueError("Event template is not initialized.")
        if not self.event_details.datetime_obj or len(self.event_details.datetime_obj.datetimes) < 1:
            raise ValueError("At least one datetime object must be provided to add an event.")
        
        try:
            if self.calendar_insights.is_event:
                for start_time, end_time in self.event_details.datetime_obj.target_datetimes:
                    self.calendar_insights.template['start']['dateTime'] = start_time
                    self.calendar_insights.template['end']['dateTime'] = end_time
                    if start_time and end_time:
                        self.calendar_insights.validate_insight_vulnerabilities()
                    self.event_service.events().insert(calendarId=self.event_id, body=self.calendar_insights.template).execute() #type: ignore
                return f"Event '{self.event_details.event_name}' added on {self.event_details.datetime_obj.target_datetimes[0]}."
            else:
                for due_time, _ in self.event_details.datetime_obj.target_datetimes:
                    local_tz = pytz.timezone('America/Los_Angeles')
                    due_datetime = local_tz.localize(datetime.fromisoformat(due_time))
                    self.calendar_insights.template['due'] = due_datetime.isoformat()
                    print(f"Due date for task '{self.event_details.event_name}': {due_datetime.isoformat()}")
                    if due_datetime:
                        self.calendar_insights.validate_insight_vulnerabilities()
                    self.task_service.tasks().insert(tasklist=self.task_list_id, body=self.calendar_insights.template).execute() #type: ignore

                return f"Task '{self.event_details.event_name}' added with due date {due_datetime}."
        except Exception as e:
            raise RuntimeError(f"An error occurred while adding the event/task: {e}")
        

#MARK: delete
class DeleteFromCalendar(HandleEvents):
    def __init__(self, event_details: EventDetails, personal_email: str = "jaceysimps@gmail.com"):
        super().__init__(event_details, personal_email)

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
                self.event_service.events().delete(calendarId=self.event_id, eventId=event_id).execute() #type: ignore
                return f"Event '{event_id}' deleted successfully."
            else:
                raise ValueError(f"No event found with ID '{event_id}'.")

        except Exception as e:
            raise RuntimeError(f"An error occurred while deleting the event/task: {e}")
        
#MARK: Update
class UpdateFromCalendar(HandleEvents):
    def __init__(self, event_details: EventDetails, personal_email: str = "jaceysimps@gmail.com"):
        super().__init__(event_details, personal_email)

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
                        original_event = self.event_service.events().get(calendarId=self.event_id, eventId=event_id).execute() #type: ignore
                        original_event['start']['dateTime'] = start_time
                        original_event['end']['dateTime'] = end_time
                        self.event_service.events().update(calendarId=self.event_id, eventId=event_id, body=original_event).execute() #type: ignore
                else:
                    for due_time, _ in event_details.datetime_obj.target_datetimes:
                        original_task = self.event_service.tasks().get(taskId=event_id).execute() #type: ignore
                        local_tz = pytz.timezone('America/Los_Angeles')
                        due_datetime = local_tz.localize(due_time)
                        original_task['due'] = due_datetime
                    self.event_service.tasks().update(taskId=event_id, body=original_task).execute() #type: ignore
                return f"Event '{event_details.event_name}' updated successfully."
            else:
                raise ValueError(f"No event found with ID '{event_id}'.")

        except Exception as e:
            raise RuntimeError(f"An error occurred while updating the event/task: {e}")
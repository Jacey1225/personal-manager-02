from src.structure_model_output import EventDetails, HandleResponse
from src.google_calendar.enable_google_api import enable_google_calendar_api
from datetime import datetime, timezone, timedelta
import pytz
from pydantic import BaseModel, Field
from typing import Optional

def check_service(event_service, task_service):
    if event_service is None:
        raise ConnectionError("Failed to connect to Google Calendar API.")
    if task_service is None:
        raise ConnectionError("Failed to connect to Google Tasks API.")
    return True

class CalendarEvent(BaseModel):
    event_name: str = Field(default="None", description="The title of the event or task")
    is_project: Optional[bool] = Field(default=False, description="Indicates that the event or task is contained within a project")
    start: datetime = Field(default=datetime.now(), description="The start datetime of the event")
    end: Optional[datetime] = Field(default=None, description="The end datetime of the event only if classified as an event")
    is_event: bool = Field(default=False, description="Determines whether we want to handle the request as an event or task")
    event_id: str = Field(default="None", description="The unique identifier for the event")

class CalendarInsights(BaseModel):
    scheduled_events: list[CalendarEvent] = Field(default=[], description="List of all processed events existing in the calendar")
    template: dict = Field(..., description="A template used to send the calendar API event info")
    is_event: bool = Field(default=False, description="Determines whether we want to handle the request as an event or task")


class HandleEvents:
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
        
#MARK: Construct Methods
        self.calendar_insights = CalendarInsights(
            scheduled_events=[],
            template={},
            is_event=False
        )
        self.fetch_events_list()
        if not self.event_details.action == "delete":
            self.fetch_targets()
        self.fetch_event_template()
             
    def fetch_events_list(self):
        """Fetch the list of events from Google Calendar and Google Tasks.

        Raises:
            ConnectionError: If unable to connect to Google Calendar or Google Tasks API.
            RuntimeError: If an error occurs while fetching events.
        """
        try:
            if check_service(self.event_service, self.task_service): #fetch existing events if needed
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
                        if event.get('start', {}).get('dateTime', '') < datetime.now(timezone.utc).isoformat():
                            continue
                        event_obj.event_name = event['summary']
                        event_obj.is_event = True
                        event_obj.start = (event.get('start', {}).get('dateTime'))
                        event_end = event.get('end', {}).get('dateTime')
                        event_obj.event_id = event.get('id')
                    elif 'title' in event:
                        if event.get('updated') < datetime.now(timezone.utc).isoformat():
                            continue
                        event_obj.event_name = event['title']
                        event_obj.start = event.get('updated')
                        event_obj.event_id = event.get('id')

                    if isinstance(event_obj.start, str):
                        event_obj.start = datetime.fromisoformat(event_obj.start)
                    if isinstance(event_end, str):
                        event_obj.end = datetime.fromisoformat(event_end)
                    
                    if not event_obj.start:
                        print(f"Event: {event}")
                        Warning("Event start time is not set.")
                        continue

                    self.calendar_insights.scheduled_events.append(event_obj)
            else:
                raise ConnectionError("Google Calendar service is not available.")
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

    def fetch_targets(self, zone:str = '-07:00'): 
        """Fetch the target dates for the event or task and set self.calendar_insights.is_event.
        """
        event_start = None
        event_end = None
        deleted_events = []
        print(f"Current Datetime Objects: {self.event_details.datetime_obj.datetimes}")
        for existing_event in self.calendar_insights.scheduled_events:
            of_interest = False
            for token in self.event_details.event_name.lower().split(" "):
                if token in existing_event.event_name.lower():
                    of_interest = True
            if not of_interest:
                print(f"Event '{existing_event.event_name}' is not of interest.")
                continue

            datetime_objs = self.event_details.datetime_obj.datetimes.copy()
            for dt in datetime_objs:
                target_time = dt.isoformat() + zone
                print(f"Comparing {target_time} with {existing_event.start} or {existing_event.end} from {existing_event.event_name}")
                if existing_event.start.isoformat() == target_time and dt not in deleted_events:
                    self.event_details.datetime_obj.datetimes.remove(dt)
                    deleted_events.append(dt)
                    print(f"Removing {dt} as it matches existing event {existing_event.start} -> {existing_event.event_name}")
                    continue
                if existing_event.end and existing_event.end.isoformat() == target_time and dt not in deleted_events:
                    self.event_details.datetime_obj.datetimes.remove(dt)
                    deleted_events.append(dt)
                    print(f"Removing {dt} as it matches existing event {existing_event.end} -> {existing_event.event_name}")
                    continue

        if len(self.event_details.datetime_obj.datetimes) < 2:
            self.calendar_insights.is_event = False
        else:
            self.calendar_insights.is_event = True

        if self.calendar_insights.is_event:
            event_start = min(self.event_details.datetime_obj.datetimes)
            event_end = max(self.event_details.datetime_obj.datetimes)
            print(f"Setting event start to {event_start} and end to {event_end}")
            self.event_details.datetime_obj.target_datetimes = (event_start, event_end)
        else:
            event_start = min(self.event_details.datetime_obj.datetimes)
            print(f"Setting task due to {event_start}")
            self.event_details.datetime_obj.target_datetimes = (event_start, None)

#MARK: add
    
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
                self.calendar_insights.template['start']['dateTime'] = self.event_details.datetime_obj.target_datetimes[0].isoformat()
                self.calendar_insights.template['end']['dateTime'] = self.event_details.datetime_obj.target_datetimes[1].isoformat()
                self.event_service.events().insert(calendarId=self.event_id, body=self.calendar_insights.template).execute() #type: ignore
                return f"Event '{self.event_details.event_name}' added on {self.event_details.datetime_obj.target_datetimes[0]}."
            else:
                due_date = self.event_details.datetime_obj.target_datetimes[0]
                local_tz = pytz.timezone('America/Los_Angeles')
                due_datetime = local_tz.localize(due_date)
                self.calendar_insights.template['due'] = due_datetime.isoformat()

                self.task_service.tasks().insert(tasklist=self.task_list_id, body=self.calendar_insights.template).execute() #type: ignore
                return f"Task '{self.event_details.event_name}' added with due date {due_datetime}."
        except Exception as e:
            raise RuntimeError(f"An error occurred while adding the event/task: {e}")
        
    def find_matching_events(self) -> list:
        matching_events = []
        for token in self.event_details.event_name.lower().split(" "):
            for event in self.calendar_insights.scheduled_events:
                if token in event.event_name.lower():
                    matching_events.append(event)
        return matching_events

#MARK: delete

    def delete_event(self):
        """Delete an event from Google Calendar or a task from Google Tasks.

        Raises:
            ValueError: If no event is found to delete.
            RuntimeError: If an error occurs while deleting the event/task.

        Returns:
            str: A message indicating the result of the operation.
        """
        try:
            if not self.calendar_insights.scheduled_events:
                print(f"No scheduled events found for '{self.event_details.event_name}'.")
                return

            matching_events = self.find_matching_events()
            print(f"Select which event you would like to remove: ")
            if not matching_events: 
                print(f"No matching events found for '{self.event_details.event_name}'.")
                print("Here are all known events:")
                for event in self.calendar_insights.scheduled_events:
                    print(f"ID: {event.event_id} | {event.event_name.upper()} | Due: {event.start} to {event.end}")
            else:
                for event in matching_events:
                    if event.end:
                        print(f"ID: {event.event_id} | {event.event_name.upper()} | Due: {event.start} to {event.end}")
                    else:
                        print(f"ID: {event.event_id} | {event.event_name.upper()} | Due: {event.start}")

            get_id = input("Enter the ID of the event you want to delete: ")
            event_to_delete = next((event for event in matching_events if event.event_id == get_id), None)
            if event_to_delete:
                self.event_service.events().delete(calendarId=self.event_id, eventId=event_to_delete.event_id).execute() #type: ignore
                return f"Event '{event_to_delete.event_name}' deleted successfully."
            else:
                raise ValueError(f"No event found with ID '{get_id}'.")

        except Exception as e:
            raise RuntimeError(f"An error occurred while deleting the event/task: {e}")

#MARK: update
    
    def update_event(self):
        """Update an existing event in Google Calendar or a task in Google Tasks.

        Raises:
            ValueError: If event name or personal email is not provided.
            RuntimeError: If an error occurs while updating the event/task.

        Returns:
            str: A message indicating the result of the operation.
        """
        if not self.event_details.datetime_obj.target_datetimes:
            raise ValueError("At least one datetime object must be provided to update an event.")
        
        try:
            matching_events = self.find_matching_events()
            if not matching_events:
                return f"No events match: {self.event_details.event_name}"
        
            print(f"Select which event you would like to update: ")
            for event in matching_events:
                print(f"ID: {event.event_id} | {event.event_name.upper()} | Due: {event.start} to {event.end}")

            get_id = input("Enter the ID of the event you want to update: ")
            event_to_update = next((event for event in matching_events if event.event_id == get_id), None)
            if event_to_update:
                if event_to_update.is_event:
                    original_event = self.event_service.events().get(calendarId=self.event_id, eventId=event_to_update.event_id).execute() #type: ignore
                    original_event['start']['dateTime'] = self.event_details.datetime_obj.target_datetimes[0].isoformat()
                    original_event['end']['dateTime'] = self.event_details.datetime_obj.target_datetimes[1].isoformat()
                    self.event_service.events().update(calendarId=self.event_id, eventId=event_to_update.event_id, body=original_event).execute() #type: ignore
                else:
                    original_task = self.event_service.tasks().get(taskId=event_to_update.event_id).execute() #type: ignore
                    due_date = self.event_details.datetime_obj.target_datetimes[0]
                    local_tz = pytz.timezone('America/Los_Angeles')
                    due_datetime = local_tz.localize(due_date)
                    original_task['due'] = due_datetime.isoformat()
                    self.event_service.tasks().update(taskId=event_to_update.event_id, body=original_task).execute() #type: ignore
                return f"Event '{event_to_update.event_name}' updated successfully."
            else:
                raise ValueError(f"No event found with ID '{get_id}'.")

        except Exception as e:
            raise RuntimeError(f"An error occurred while updating the event/task: {e}")
        
if __name__ == "__main__":
    input_text = "Can you actually move my workout session tomorrow from 9:30 PM to 10:00 PM to 8:00 PM to 9:30 PM?"
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
        print(result)

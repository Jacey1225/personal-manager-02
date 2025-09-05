from src.structure_model_output import EventDetails, HandleResponse
from src.google_calendar.enable_google_api import enable_google_calendar_api
from datetime import datetime, timedelta
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
    duration: Optional[timedelta] = Field(default=timedelta(hours=1), description="The duration of the event")
    is_event: bool = Field(default=False, description="Determines whether we want to handle the request as an event or task")

class CalendarInsights(BaseModel):
    scheduled_events: list[CalendarEvent] = Field(default=[], description="List of all processed events existing in the calendar")
    template: dict = Field(..., description="A template used to send the calendar API event info")
    is_event: bool = Field(default=False, description="Determines whether we want to handle the request as an event or task")


class HandleEvents:
    def __init__(self, event_details: EventDetails, personal_email: str = "jaceysimps@gmail.com"):
        self.event_service, self.task_service = enable_google_calendar_api()
        self.personal_email = personal_email
        self.event_details = event_details
        self.task_list_id = '@default'
        self.event_id = 'primary'

        if not self.event_details.event_name or not self.event_details.action:
            print(f"Event Name: {self.event_details.event_name}")
            print(f"Event Action: {self.event_details.action}")
            print(f"Input Text: {self.event_details.input_text}")
            raise ValueError("Event name and action must be provided in event_details.")
        if len(self.event_details.datetime_objs) < 1:
            raise ValueError("At least one datetime object must be provided in event_details.")
        
# Event Handling Setup
        self.calendar_insights = CalendarInsights(
            scheduled_events=[],
            template={},
            is_event=False
        )
        self.fetch_events_list()
        self.verify_event()
        self.fetch_event_template()

    def verify_event(self):
        if self.event_details.action.lower() == "add" and len(self.event_details.datetime_objs) > 1:
            self.calendar_insights.is_event = True
        if self.calendar_insights.scheduled_events:
            for scheduled in self.calendar_insights.scheduled_events:
                if scheduled.event_name.lower() == self.event_details.event_name.lower() and scheduled.is_event:
                    self.calendar_insights.is_event = True
                    break
                if scheduled.event_name.lower() == self.event_details.event_name.lower() and not scheduled.is_event:
                    self.calendar_insights.is_event = False
                    break
        return self.calendar_insights.is_event
    
    def fetch_events_list(self):
        try:
            request_body = {
                'maxResults': 10,
            }
            if check_service(self.event_service, self.task_service): #fetch existing events if needed
                tasks_list = self.task_service.tasks().list(tasklist=self.task_list_id, **request_body).execute() #type: ignore
                events_list = self.event_service.events().list(calendarId=self.event_id, **request_body).execute() #type: ignore
                scheduled_events = events_list.get('items', []) + tasks_list.get('items', []) #type: ignore
                for event in scheduled_events:
                    event_obj = CalendarEvent(
                        event_name='',
                        start=datetime.now(),
                        duration=None,
                        is_event=False
                    )
                    if 'summary' in event:
                        event_obj.event_name = event['summary']
                        event_obj.is_event = True
                        event_obj.start = (event.get('start', {}).get('dateTime'))
                        event_end = event.get('end', {}).get('dateTime')
                    elif 'title' in event:
                        event_obj.event_name = event['title']
                        event_obj.start = event.get('due')
                        event_obj.duration = timedelta(hours=1)

                    if isinstance(event_obj.start, str):
                        event_obj.start = datetime.fromisoformat(event_obj.start)
                    if isinstance(event_end, str):
                        event_end = datetime.fromisoformat(event_end)
                    if not event_obj.duration and event_end:
                        event_obj.duration = (event_end - event_obj.start)

                    self.calendar_insights.scheduled_events.append(event_obj)
            else:
                raise ConnectionError("Google Calendar service is not available.")
        except AttributeError as e:
            print(f"Service attribute error: {e}")
            pass
        except Exception as e:
            raise RuntimeError(f"An error occurred while fetching events: {e}")
    
    def fetch_event_template(self):
        if not self.event_details.event_name or not self.personal_email:
            raise ValueError("Event name and personal email must be provided.")
        if not self.calendar_insights.is_event:
            self.calendar_insights.template = {
                'title': self.event_details.event_name,
                'notes': '',
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
# --------------------------------------------------------------------- Constructor Methods --------------------------------------------------------------------- #    
    def sort_due_dates(self): #ensure we get the right due date if multiple dates were found
        event_start = None
        event_duration = None       
        for possible_time in self.event_details.datetime_objs:
            for existing_event in self.calendar_insights.scheduled_events:
                if possible_time != existing_event.start:
                    event_start = possible_time
                    event_duration = existing_event.duration

        if not event_start and not event_duration:
            if len(self.event_details.datetime_objs) > 1:
                event_start = min(self.event_details.datetime_objs)
                event_end = max(self.event_details.datetime_objs)
                event_duration = event_end - event_start
            else:
                event_start = self.event_details.datetime_objs[0]
                event_duration = timedelta(hours=1)

        print(f"Event Start: {event_start}, Event Duration: {event_duration}")
        return event_start, event_duration

#MARK: add
    
    def add_event(self):
        if not self.calendar_insights.template:
            raise ValueError("Event template is not initialized.")
        if not self.event_details.datetime_objs or len(self.event_details.datetime_objs) < 1:
            raise ValueError("At least one datetime object must be provided to add an event.")
        
        try:
            if self.calendar_insights.is_event:
                target_start, duration = self.sort_due_dates()
                if not duration or not target_start:
                    print(f"Event Details: {self.event_details}")
                    raise ValueError("Both start time and duration must be determined to add an event.")
                target_end = target_start + duration
                self.calendar_insights.template['start']['dateTime'] = target_start.isoformat()
                self.calendar_insights.template['end']['dateTime'] = target_end.isoformat()
                event = self.event_service.events().insert(calendarId=self.event_id, body=self.calendar_insights.template).execute() #type: ignore
                return f"Event '{self.event_details.event_name}' added on {target_start.strftime('%Y-%m-%d %H:%M')}."
            else:
                target_due, _ = self.sort_due_dates()
                self.calendar_insights.template['due'] = target_due.isoformat() #type: ignore
                task = self.task_service.tasks().insert(tasklist=self.task_list_id, body=self.calendar_insights.template).execute() #type: ignore
                return f"Task '{self.event_details.event_name}' added with due date {target_due}."
        except Exception as e:
            raise RuntimeError(f"An error occurred while adding the event/task: {e}")
        
#MARK: delete
    
    def delete_event(self):
        try:
            if self.calendar_insights.is_event:
                events = self.event_service.events().list(calendarId=self.event_id, q=self.event_details.event_name).execute() #type: ignore
                for event in events.get('items', []):
                    if event['summary'] == self.event_details.event_name:
                        self.event_service.events().delete(calendarId=self.event_id, eventId=event['id']).execute() #type: ignore
                        return f"Event '{self.event_details.event_name}' deleted."
                return f"No matching event found for '{self.event_details.event_name}'."
            else:
                tasks = self.task_service.tasks().list(tasklist=self.task_list_id).execute() #type: ignore
                for task in tasks.get('items', []):
                    if task['title'] == self.event_details.event_name:
                        self.task_service.tasks().delete(tasklist=self.task_list_id, task=task['id']).execute() #type: ignore
                        return f"Task '{self.event_details.event_name}' deleted."
                return f"No matching task found for '{self.event_details.event_name}'."
        except Exception as e:
            raise RuntimeError(f"An error occurred while deleting the event/task: {e}")

#MARK: update
    
    def update_event(self):
            
        if not self.event_details.datetime_objs or len(self.event_details.datetime_objs) < 1:
            raise ValueError("At least one datetime object must be provided to update an event.")
        
        try:
            if self.calendar_insights.is_event:
                events = self.event_service.events().list(calendarId=self.event_id, q=self.event_details.event_name).execute() #type: ignore
                for event in events.get('items', []):
                    if event['summary'] == self.event_details.event_name:
                        target_start, duration = self.sort_due_dates()
                        if not duration:
                            duration = timedelta(hours=1) #default to 1 hour if no duration found
                        target_end = target_start + duration #type: ignore
                        event['start']['dateTime'] = target_start.isoformat() #type: ignore
                        event['end']['dateTime'] = target_end.isoformat()
                        updated_event = self.event_service.events().update(calendarId=self.event_id, eventId=event['id'], body=event).execute() #type: ignore
                        return f"Event '{self.event_details.event_name}' updated to {target_start}."
                return f"No matching event found for '{self.event_details.event_name}'."
            else:
                tasks = self.task_service.tasks().list(tasklist=self.task_list_id).execute() #type: ignore
                for task in tasks.get('items', []):
                    if task['title'] == self.event_details.event_name:
                        target_due, _ = self.sort_due_dates()
                        task['due'] = target_due.isoformat() #type: ignore
                        updated_task = self.task_service.tasks().update(tasklist=self.task_list_id, task=task['id'], body=task).execute() #type: ignore
                        return f"Task '{self.event_details.event_name}' updated with due date {target_due}."
                return f"No matching task found for '{self.event_details.event_name}'."
        except Exception as e:
            raise RuntimeError(f"An error occurred while updating the event/task: {e}")
        
if __name__ == "__main__":
    input_text = "Can you schedule my work today that's at 4:00pm?"
    if not "." in input_text or not "?" in input_text:
        input_text += "."
    response_handler = HandleResponse(input_text)
    events = response_handler.process_response()
    print(f"Found Events: {events}")
    for event in events:
        calendar_handler = HandleEvents(event)
        if event.action.lower() == "add":
            result = calendar_handler.add_event()
        elif event.action.lower() == "delete":
            result = calendar_handler.delete_event()
        elif event.action.lower() == "update":
            result = calendar_handler.update_event()
        else:
            result = f"Unknown action '{event.action}' for event '{event.event_name}'."
        print(result)

from typing import Optional
from pydantic import BaseModel, Field
from api.schemas.calendar import CalendarEvent
import keyring

class FormattedAppleEvent(BaseModel):
    title: str = Field(description="The title of the Apple Calendar event")
    start_dt: datetime = Field(description="The start date and time of the Apple Calendar event")
    end_dt: datetime = Field(description="The end date and time of the Apple Calendar event")
    pguid: str = Field(description="The GUID of the Apple Calendar event")

class UniversalServices:
    def __init__(self, 
                google_calendar_name: Optional[str] = None,
                google_events = None,
                google_tasks_name: Optional[str] = None,
                google_tasks = None,
                apple_calendar_name: Optional[str] = None,
                apple_events = None
                 ):
        #API Configuration params
        self.google_calendar_name = google_calendar_name
        self.google_events = google_events

        self.google_tasks_name = google_tasks_name
        self.google_tasks = google_tasks

        self.apple_calendar_name = apple_calendar_name
        self.apple_events = apple_events

    def fetch_event(self, **kwargs):
        """Fetches a specific event from the user's calendar and task services.

        Args:
            **kwargs: Additional parameters to filter the event.

        Returns:
            Optional[dict]: The fetched event data or None if not found.
        """
        try:
            if self.google_tasks and kwargs.get('task_id'):
                return self.google_tasks.tasks().get(
                    tasklist=self.google_tasks_name,
                    taskId=kwargs.get('task_id')
                ).execute()
        except Exception as e:
            print(f"Error fetching google task: {e}")
            return None

        try:
            if self.google_events and kwargs.get('event_id'):
                return self.google_events.events().get(
                    calendarId=self.google_calendar_name,
                    eventId=kwargs.get('event_id')
                ).execute()
        except Exception as e:
            print(f"Error fetching google event: {e}")
            return None

        try:
            if self.apple_events and kwargs.get('event_id'):
                return self.apple_events.events().get(
                    calendarId=self.apple_calendar_name,
                    eventId=kwargs.get('event_id')
                ).execute()
        except Exception as e:
            print(f"Error fetching apple event: {e}")
            return None

        return None

    def format_events(self, scheduled_events: list) -> Optional[CalendarEvent]:
        """Formats the scheduled events into CalendarEvent objects.

        Args:
            scheduled_events (list): The list of scheduled events to format.

        Yields:
            CalendarEvent: The formatted CalendarEvent object.
        """
        
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

            yield event_obj 

    def event_list(self, **kwargs) -> Optional[list]:
        """Fetches the list of events from the configured calendar and task services.

        Returns:
            Optional[list]: The list of events from the user's calendar and task services.
        """
        tasks = []
        events = []
        try:
            if self.google_tasks:
                tasks = self.google_tasks.tasks().list(
                    tasklist=self.google_tasks_name,
                    dueMin=kwargs.get('dueMin', None),
                    dueMax=kwargs.get('dueMax', None)
                ).execute()
        except Exception as e:
            print(f"Error fetching google tasks: {e}")
            return []
        
        try:
            if self.google_events:
                events.extend(self.google_events.events().list(
                    calendarId=self.google_calendar_name,
                    maxResults=kwargs.get('maxResults', 100),
                    timeMin=kwargs.get('timeMin', None),
                    timeMax=kwargs.get('timeMax', None),
                    description=kwargs.get('description', None)
                ).execute())
        except Exception as e:
            print(f"Error fetching google events: {e}")
            return []
        
        try:
            if self.apple_events:
                self.apple_events = \
                self.apple_events.calendar.calendars[self.apple_calendar_name]
                events.extend(self.apple_events.events(
                    kwargs.get('dueMin', None),
                    kwargs.get('dueMax', None)
                ).execute())
        except Exception as e:
            print(f"Error fetching apple events: {e}")
            return []

        merged = tasks.get('items', []) + events.get('items', [])
        return merged

    def add_event(self, **kwargs) -> dict:
        """Adds an event to the user's calendar and task services.

        Returns:
            dict: The status of the add event operation.
        """
        try:
            if self.google_tasks and 'due' in kwargs.get('body', {}):
                self.google_tasks.tasks().insert(
                    tasklist=self.google_tasks_name,
                    body=kwargs.get('template', None)
                ).execute()
        except Exception as e:
            print(f"Error adding google task: {e}")
            return {"status": "error", "message": str(e)}

        try:
            if self.google_events and 'start' in kwargs.get('body', {}):
                self.google_events.events().insert(
                    calendarId=self.google_calendar_name,
                    body=kwargs.get('template', None)
                ).execute()
        except Exception as e:
            print(f"Error adding google event: {e}")
            return {"status": "error", "message": str(e)}

        try:
            if self.apple_events:
                self.apple_events = \
                self.apple_events.calendar.calendars[self.apple_calendar_name]
                new_body = FormattedAppleEvent(
                    title=kwargs.get('body', {}).get('summary', ''),
                    start_dt=kwargs.get('body', {}).get('start', {}).get('dateTime', ''),
                    end_dt=kwargs.get('body', {}).get('end', {}).get('dateTime', ''),
                    pguid=self.apple_events.id
                )
                self.apple_events.add_event(
                    new_body
                )
        except Exception as e:
            print(f"Error adding apple event: {e}")
            return {"status": "error", "message": str(e)}
    
        return {"status": "success"}

    def delete_event(self, **kwargs) -> dict:
        try:
            if self.google_events:
                self.google_events.events().delete(
                    calendarId=self.google_calendar_name,
                    eventId=kwargs.get('event_id', None)
                ).execute()
        except Exception as e:
            print(f"Error deleting google event: {e}")
            return {"status": "error", "message": str(e)}

        try:
            if self.apple_events:
                self.apple_events = \
                self.apple_events.calendar.calendars[self.apple_calendar_name]
                self.apple_events.delete_event(
                    kwargs.get('event_id', None)
                )
        except Exception as e:
            print(f"Error deleting apple event: {e}")
            return {"status": "error", "message": str(e)}

        return {"status": "success"}

    def delete_task(self, **kwargs):
        try:
            if self.google_tasks:
                self.google_tasks.tasks().delete(
                    tasklist=self.google_tasks_name,
                    taskId=kwargs.get('task_id', None)
                ).execute()
        except Exception as e:
            print(f"Error deleting google task: {e}")
            return {"status": "error", "message": str(e)}

        return {"status": "success"}

    def update_event(self, **kwargs) -> dict:
        try:
            if self.google_tasks:
                self.google_tasks.tasks().update(
                    tasklist=self.google_tasks_name,
                    eventId=kwargs.get('event_id', None)
                    body=kwargs.get('body', None)
                ).execute()
        except Exception as e:
            print(f"Error updating google task: {e}")
            return {"status": "error", "message": str(e)}

        try:
            if self.google_events:
                self.google_events.events().update(
                    calendarId=self.google_calendar_name,
                    eventId=kwargs.get('event_id', None),
                    body=kwargs.get('body', None)
                ).execute()
        except Exception as e:
            print(f"Error updating google event: {e}")
            return {"status": "error", "message": str(e)}

        try:
            if self.apple_events:
                self.apple_events = \
                self.apple_events.calendar.calendars[self.apple_calendar_name]
                new_body = FormattedAppleEvent(
                    title=kwargs.get('body', {}).get('summary', ''),
                    start_dt=kwargs.get('body', {}).get('start', {}).get('dateTime', ''),
                    end_dt=kwargs.get('body', {}).get('end', {}).get('dateTime', ''),
                )
                self.apple_events.update_event(
                    kwargs.get('event_id', None),
                    new_body
                )
        except Exception as e:
            print(f"Error updating apple event: {e}")
            return {"status": "error", "message": str(e)}

        return {"status": "success"}

import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build 
from api.schemas.calendar import CalendarEvent
from typing import Optional
from api.schemas.calendar import CalendarEvent
from api.config.fetchMongo import MongoHandler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Updated scopes - removed conflicting readonly scopes
SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly", 
    "https://www.googleapis.com/auth/tasks.readonly", 
    "https://www.googleapis.com/auth/tasks", 
    "https://www.googleapis.com/auth/calendar.events"
]

class ConfigureGoogleAPI:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.flow = None
        self.creds = None
        self.auth_url = None
        if not os.path.exists('data/tokens'):
            os.makedirs('data/tokens')

    async def write_token(self):
        """Writes a token for the specified user and updates the database with it

        Returns:
            _type_: _description_
        """
        user_config = MongoHandler("userAuthDatabase", "userCredentials")
        if self.creds:
            await user_config.post_update({"user_id": self.user_id}, {"google_auth": self.creds.to_json()})
            logger.info(f"Token written to database for user {self.user_id}")
            return True
        else:
            logger.warning(f"No credentials available to write for user {self.user_id}")
            return False

    def get_auth_url(self):
        """Generate and return the Google OAuth authorization URL"""
        if not os.path.exists('data/credentials.json'):
            raise FileNotFoundError("credentials.json file not found in the 'data' directory.")
        
        self.flow = InstalledAppFlow.from_client_secrets_file('data/credentials.json', SCOPES)
        self.flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
        auth_url, _ = self.flow.authorization_url(prompt='consent', access_type='offline')
        self.auth_url = auth_url
        return self.auth_url

    async def complete_auth_flow(self, authorization_code: str):
        """Complete OAuth flow with authorization code"""
        logger.info(f"Starting complete_auth_flow for user {self.user_id}")
        logger.info(f"Authorization code provided: {len(authorization_code)} characters")
        try:
            if not os.path.exists('data/credentials.json'):
                raise FileNotFoundError("credentials.json file not found in the 'data' directory.")

            logger.info("Creating fresh OAuth flow for token exchange...")
            self.flow = InstalledAppFlow.from_client_secrets_file('data/credentials.json', SCOPES)
            self.flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"

            logger.info("Exchanging authorization code for token...")

            self.flow.fetch_token(code=authorization_code)
            self.creds = self.flow.credentials

            logger.info("Credentials obtained successfully")
            logger.info(f"Credentials valid: {self.creds.valid}")
            logger.info(f"Has refresh token: {self.creds.refresh_token is not None}")

            logger.info("Writing token to database...")
            token_written = await self.write_token()

            if not token_written:
                logger.error(f"Failed to write token to database for user {self.user_id}")
                return None, None

            logger.info("Testing credentials by building Google services...")
            event_service = build('calendar', 'v3', credentials=self.creds)
            task_service = build('tasks', 'v1', credentials=self.creds)
            
            print("Google services built successfully!")
            return event_service, task_service
            
        except Exception as e:
            print(f"ERROR in complete_auth_flow: {e}")
            print(f"Exception type: {type(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    async def enable(self) -> tuple | str | None:
        """Enable Google Calendar API - returns services if authenticated, auth URL if not"""
        
        user_config = MongoHandler("userAuthDatabase", "userCredentials")
        user_info = await user_config.get_single_doc({"user_id": self.user_id})

        if user_info:
            self.creds = Credentials.from_authorized_user_info(user_info, SCOPES)

            try:
                if self.creds and self.creds.valid:
                    event_service = build('calendar', 'v3', credentials=self.creds)
                    task_service = build('tasks', 'v1', credentials=self.creds)
                    return event_service, task_service
                
                elif self.creds and self.creds.expired and self.creds.refresh_token:
                    try:
                        self.creds.refresh(Request())
                        await self.write_token()
                        event_service = build('calendar', 'v3', credentials=self.creds)
                        task_service = build('tasks', 'v1', credentials=self.creds)
                        return event_service, task_service
                    except Exception as e:
                        logger.error(f"Token refresh failed: {e}")
                        return self.get_auth_url()
                else:
                    return self.get_auth_url()
                    
            except Exception as e:
                logger.error(f"Error loading token: {e}")
                return self.get_auth_url()
        else:
            return self.get_auth_url()
        
class SyncGoogleEvents:
    def __init__(self, calendar: str, event_service):
        self.service = event_service
        self.calendar = calendar

    def template(self, **kwargs) -> dict:
        template = {
                "summary": kwargs.get("event_name", None),
                "description": kwargs.get("description", None),
                "start": {
                    "dateTime": kwargs.get("start", None),
                    "timeZone": kwargs.get("timezone", "America/Los_Angeles")
                },
                "end": {
                    "dateTime": kwargs.get("end", None),
                    "timeZone": kwargs.get("timezone", "America/Los_Angeles")
                },
                "attendees": [{"email": email} for email in kwargs.get("attendees", [])],
                "transparency": kwargs.get("transparency", "opaque"),
                "guestsCanModify": kwargs.get("guestCanModify", False)
            }
        return template

    def format(self, event: dict) -> Optional[CalendarEvent]:
        """Formats an event from the Google Calendar API.

        Args:
            event (dict): The event data from the Google Calendar API.

        Returns:
            Optional[CalendarEvent]: CalendarEvent object with formatted event data.
        """
        try:
            calendar_event = CalendarEvent()
            calendar_event.event_name = event.get("summary", "")
            calendar_event.event_id = event.get("id", None)
            calendar_event.description = event.get("description", "")
            calendar_event.start = event.get("start", {}).get('dateTime', None)
            calendar_event.end = event.get("end", {}).get('dateTime', None)
            calendar_event.timezone = event.get("start", {}).get('timeZone', "America/Los_Angeles")
            calendar_event.status = event.get("status", "unknown")
            calendar_event.transparency = event.get("transparency", "opaque")
            calendar_event.attendees = [attendee.get("email") for attendee in event.get("attendees", [])]
            calendar_event.guestsCanModify = event.get("guestsCanModify", False)
            return calendar_event
        except Exception as e:
            print(f"Error formatting event: {e}")
            return CalendarEvent()
        
    def list_events(self, **kwargs) -> list[Optional[CalendarEvent]]:
        """Lists all events in the calendar.

        Returns:
            list[CalendarEvent]: A list of CalendarEvent objects.
        """
        events = self.service.events().list(
            calendarId=self.calendar,
            maxResults=kwargs.get("maxResults"),
            timeMin=kwargs.get("timeMin"),
            timeMax=kwargs.get("timeMax"),
            q=kwargs.get("description")
        ).execute().get('items', [])
        try:
            return [self.format(event) for event in events]
        except Exception as e:
            print(f"Error listing events: {e}")
            return []

    def fetch_event(self, **kwargs) -> CalendarEvent | None:
        """Fetches a specific event from the calendar.

        Args:
            event_id (str): The ID of the event to fetch.

        Returns:
            CalendarEvent | None: The CalendarEvent object if found, else None.
        """
        event = self.service.events().get(
            calendarId=self.calendar, 
            eventId=kwargs.get("event_id")).execute()
        return self.format(event)
    
    def create_event(self, template: dict) -> CalendarEvent | None:
        """Creates a new event in the calendar.

        Args:
            event (CalendarEvent): The event to create.

        Returns:
            CalendarEvent | None: The created CalendarEvent object if successful, else None.
        """
        created_event = self.service.events().insert(
            calendarId=self.calendar, 
            body=template).execute()
        return self.format(created_event)

    def delete_event(self, **kwargs) -> dict | None:
        """Deletes an event from the calendar.

        Args:
            event_id (str): The ID of the event to delete.

        Returns:
            dict | None: A success message if deleted, else None.
        """
        self.service.events().delete(
            calendarId=self.calendar, 
            eventId=kwargs.get("event_id")).execute()
        return {"status": "success"}

    def update_event(self, event: CalendarEvent) -> Optional[CalendarEvent]:
        """Updates an existing event in the calendar.

        Args:
            event (CalendarEvent): The event to update.

        Returns:
            CalendarEvent | None: The updated CalendarEvent object if successful, else None.
        """
        event_to_update = event.to_google()
        self.service.events().update(
            calendarId=self.calendar, 
            eventId=event.event_id, 
            body=event_to_update).execute()
        return event

class SyncGoogleTasks:
    def __init__(self, calendar, task_service):
        self.task_service = task_service
        self.calendar = calendar

    def template(self, **kwargs):
        template = {
            "title": kwargs.get("event_name", None),
            "due": {
                "dateTime": kwargs.get("start", None),
                "timeZone": kwargs.get("timezone", "America/Los_Angeles")
            },
            "notes": kwargs.get("description", None),
            "status": kwargs.get("status", "needsAction"),
        }
        return template

    def format(self, task) -> Optional[CalendarEvent]:
        """Formats a task from the Google Tasks API.

        Args:
            task (dict): The task data from the Google Tasks API.

        Returns:
            CalendarEvent: CalendarEvent object with formatted task data.
        """
        try:
            calendar_task = CalendarEvent()
            calendar_task.event_name = task.get("title", "")
            calendar_task.start = (task.get("due", {}).get('dateTime'))
            calendar_task.description = task.get("notes", None)
            calendar_task.timezone = task.get('due', {}).get('timeZone', 'America/Los_Angeles')
            calendar_task.event_id = task.get("id", None)
            return calendar_task
        except Exception as e:
            print(f"Error formatting task: {e}")
            return None

    def list_events(self, **kwargs) -> list[Optional[CalendarEvent]]:
        """Lists all events in the calendar.

        Returns:
            list[CalendarEvent]: A list of CalendarEvent objects.
        """
        tasks = self.task_service.tasks().list(
            tasklist=self.calendar,
            dueMin=kwargs.get("minTime"),
            dueMax=kwargs.get("maxTime")).execute().get('items', [])
        
        task_list = []
        for task in tasks:
            task_dict = self.format(task)
            description = kwargs.get("description", None)
            if task_dict and description and description not in task_dict.description:
                continue
            task_list.append(task_dict)
        return task_list

    def fetch_event(self, **kwargs) -> CalendarEvent | None:
        """Fetches a specific event from the calendar.

        Args:
            event_id (str): The ID of the event to fetch."""

        task = self.task_service.tasks().get(
            tasklist=self.calendar, 
            task=kwargs.get("event_id")).execute()
        return self.format(task)    
    def create_event(self, template: dict) -> CalendarEvent | None:
        """Creates a new event in the calendar.

        Args:
            event (CalendarEvent): The event to create.

        Returns:
            CalendarEvent | None: The created CalendarEvent object if successful, else None.
        """
        created_task = self.service.tasks().insert( #type: ignore
            tasklist=self.calendar, 
            body=template).execute()
        return self.format(created_task)

    def delete_event(self, **kwargs) -> dict | None:
        """Deletes an event from the calendar.

        Args:
            event_id (str): The ID of the event to delete.

        Returns:
            dict | None: A success message if deleted, else None.
        """
        self.service.tasks().delete( #type: ignore
            tasklist=self.calendar, 
            eventId=kwargs.get("event_id")).execute()
        return {"status": "success"}

    def update_event(self, event: CalendarEvent) -> CalendarEvent | None:
        """Updates an existing event in the calendar.

        Args:
            event (CalendarEvent): The event to update.

        Returns:
            CalendarEvent | None: The updated CalendarEvent object if successful, else None.
        """
        event_to_update = event.to_google()
        self.service.tasks().update( #type: ignore
            tasklist=self.calendar, 
            eventId=event.event_id, 
            body=event_to_update).execute()
        return self.format(event)

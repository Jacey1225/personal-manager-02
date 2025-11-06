from datetime import datetime, timedelta
import keyring
from api.schemas.calendar import CalendarEvent
from pyicloud import PyiCloudService
from typing import Optional

class ConfigureAppleAPI:
    def __init__(self,
                 user_id: str,
                 ):
        self.user_id = user_id
        self.icloud_user = None
    
    def fetch_icloud_user(self) -> Optional[str]:
        """Fetches the iCloud username from MongoDB userCredentials.

        Returns:
            Optional[str]: The iCloud username if found, else None.
        """
        pass

    def enable(self, auth_code: str) -> Optional[PyiCloudService]:
        try:
            if not self.icloud_user:
                print("iCloud user not found")
                return None
            
            icloud_pass = keyring.get_password(
                "user_auth",
                self.icloud_user
            )
            if not icloud_pass:
                print("iCloud password not found")
                return None
            else:
                api = PyiCloudService(self.icloud_user, icloud_pass)
                if api.requires_2fa:
                    result = api.validate_2fa_code(auth_code)
                    if not result:
                        print("Failed to verify 2FA code")
                        return None
                return api
        except Exception as e:
            print(f"Error fetching user auth: {str(e)}")
            return None

class SyncAppleEvents:
    def __init__(self, calendar_name: str, service: PyiCloudService):
        self.calendar = calendar_name
        self.service = service.calendar

    def template(self, **kwargs) -> dict:
        template = {
            "pguid": self.calendar,
            "guid": kwargs.get("event_id", None),
            "title": kwargs.get("event_name", None),
            "startDate": kwargs.get("start", None),
            "endDate": kwargs.get("end", None),
            "status": kwargs.get("status", "pending"),
            "transparency": kwargs.get("transparency", "transparent"),
            "location": kwargs.get("timezone", "America/Los_Angeles")
        }
        return template

    def format(self, event: dict) -> Optional[CalendarEvent]:
        """Formats an event from the Apple Calendar API.

        Args:
            event (dict): The event data from the Apple Calendar API.

        Returns:
            CalendarEvent: CalendarEvent object with formatted event data.
        """
        try:
            calendar_event = CalendarEvent()
            calendar_event.calendar_id = event.get('pguid', None)
            calendar_event.event_id = event.get('guid', None) 
            calendar_event.event_name = event.get('title', "No Title")
            calendar_event.start = event.get('startDate', datetime.now().isoformat())
            calendar_event.end = event.get('endDate', (datetime.now() + timedelta(hours=1)).isoformat())
            calendar_event.status = event.get('status',  "COMPLETE")
            calendar_event.description = event.get('description', '')
            return calendar_event
        except Exception as e:
            print(f"Error formatting event: {str(e)}")
            return CalendarEvent()
        
    def list_events(self, **kwargs) -> list[Optional[CalendarEvent]]:
        """Lists all events in the Apple Calendar.

        Returns:
            list[CalendarEvent]: A list of CalendarEvent objects.
        """
        try:
            events = self.service.get_events(
                kwargs.get("timeMin"),
                kwargs.get("timeMax")
            )
            event_list = [self.format(event) for event in events]
            return event_list   
        except Exception as e:
            print(f"Error listing events: {str(e)}")
            return []

    def fetch_event(self, **kwargs) -> Optional[CalendarEvent]:
        """Fetches a specific event from the Apple Calendar.

        Args:
            event_id (str): The unique identifier for the event.

        Returns:
            CalendarEvent | None: The CalendarEvent object if found, else None.
        """
        try:
            event = self.service.get_event_detail(self.calendar, kwargs.get("event_id"))
            return self.format(vars(event))
        except Exception as e:
            print(f"Error fetching event: {str(e)}")
            return None

    def create_event(self, template: dict) -> CalendarEvent | None:
        """Creates a new event in the Apple Calendar.

        Args:
            event (CalendarEvent): The CalendarEvent object to create.

        Returns:
            CalendarEvent | None: The created CalendarEvent object if successful, else None.
        """
        try:
            self.service.add_event(**template)
            return self.format(template)
        except Exception as e:
            print(f"Error creating event: {str(e)}")
            return None

    def delete_event(self, **kwargs) -> dict | None:
        """Deletes an event from the Apple Calendar.

        Args:
            event_id (str): The unique identifier for the event.

        Returns:
            dict | None: A success message if deleted, else None.
        """
        try:
            event = self.service.get_event_detail(self.calendar, kwargs.get("event_id"))
            self.service.remove_event(event)
            return {"status": "success"}
        except Exception as e:
            print(f"Error deleting event: {str(e)}")
            return None

    def update_event(self, event: CalendarEvent) -> CalendarEvent | None:
        """Updates an existing event in the Apple Calendar.

        Args:
            event (CalendarEvent): The CalendarEvent object with updated data.

        Returns:
            CalendarEvent | None: The updated CalendarEvent object if successful, else None.
        """
        try:
            target_event = self.service.get_event_detail(self.calendar, event.event_id)
            updated_event = event.to_apple()
            self.service.remove_event(target_event)
            self.service.add_event(**updated_event)
        except Exception as e:
            print(f"Error updating event: {str(e)}")
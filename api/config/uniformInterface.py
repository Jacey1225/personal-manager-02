from typing import Optional
from api.schemas.calendar import CalendarEvent
from api.config.fetchMongo import MongoHandler, MongoClient

user_config = MongoHandler(None, "userAuthDatabase", "userCredentials")

"""Accepted Kwargs:
    maxResults: int
    timeMin: datetime
    timeMax: datetime
    event_id: str
    is_event: bool

    event_name: str
    start: datetime
    end: datetime
    description: str
    status: str
    transparency: str
    guestCanModify: bool
    attendees: list[str]
    timezone: str
    """
class UniformInterface:
    def __init__(self, user_id: str):
        self.user_id = user_id

    async def fetch_service(self) -> Optional[object]:
        """Fetches the services for the user.

        Yields:
            service instances for each enabled calendar service.
        """
        await user_config.get_client()
        if not user_config.client:
            raise ConnectionError("Unable to connect to user configuration database.")
        
        user_info = await user_config.get_single_doc({"user_id": self.user_id})
        if not user_info:
            print("User configuration not found.")
            return None
        service = user_info.get("service", {})
        config_ref = service.get("Configuration", None)
        service_ref = service.get("Interface", None)
        if config_ref and service_ref:
            configuration = getattr(self, config_ref)
            api = configuration(self.user_id).enable()
            service = getattr(self, service_ref)
            service_instance = service(self.user_id, api)
            return service_instance
        else:
            print("Invalid service configuration.")
            return None

    def format(self, event: dict) -> Optional[CalendarEvent]:
        """Formats an event from the calendar API.

        Args:
            event (dict): The event data from the calendar API.

        Returns:
            Optional[CalendarEvent]: CalendarEvent object with formatted event data.
        """
        pass
    def list_events(self) -> list[Optional[CalendarEvent]]:
        """Lists all events in the calendar.

        Returns:
            list[Optional[CalendarEvent]]: A list of CalendarEvent objects.
        """
        return []
    
    def fetch_event(self, event_id: str) -> Optional[CalendarEvent]:
        """Fetches a specific event from the calendar.

        Args:
            event_id (str): The ID of the event to fetch.

        Returns:
            CalendarEvent | None: The CalendarEvent object if found, else None.
        """
        pass
    def create_event(self, event: CalendarEvent) -> Optional[CalendarEvent]:
        """Creates a new event in the calendar.

        Args:
            event (CalendarEvent): The event to create.

        Returns:
            CalendarEvent | None: The created CalendarEvent object if successful, else None.
        """
        pass
    def delete_event(self, event_id: str) -> Optional[dict]:
        """Deletes an event from the calendar.

        Args:
            event_id (str): The ID of the event to delete.

        Returns:
            dict | None: A success message if deleted, else None.
        """
        pass
    def update_event(self, event: CalendarEvent) -> Optional[CalendarEvent]:
        """Updates an existing event in the calendar.

        Args:
            event (CalendarEvent): The event to update.

        Returns:
            CalendarEvent | None: The updated CalendarEvent object if successful, else None.
        """
        pass
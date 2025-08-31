import enable_api

class HandleEvents:
    def __init__(self, personal_email):
        self.service = enable_api.enable_google_calendar_api()
        self.personal_email = personal_email
        if not self.service:
            raise ConnectionError("Failed to connect to Google Calendar API.")

    def create_event(self, event_details):
        if event_details["event_name"] == "None" or event_details["event_date"] == "None" or event_details["event_time"] == "None":
            raise ValueError("Event name, date, and time must be provided to create an event.")
        event = {
            "summary": event_details["event_name"],
            "start": {
                "dateTime": f"{event_details['event_date']}T{event_details['event_time']}:00",
                "timeZone": "America/Los_Angeles",
            },
            "end": {
                "dateTime": f"{event_details['event_date']}T{event_details['event_time']}:00",
                "timeZone": "America/Los_Angeles",
            },
            "attendees": [
                {"email": self.personal_email}
                ],
        }
        try:
            event_result = self.service.events().insert(calendarId='primary', body=event, sendUpdates='all').execute() #type: ignore
            return event_result
        except Exception as e:
            raise RuntimeError(f"An error occurred while creating the event: {e}")
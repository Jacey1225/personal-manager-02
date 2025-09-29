from src.google_calendar.eventSetup import RequestSetup
from src.model_setup.structure_model_output import EventDetails
from pydantic import BaseModel
from src.google_calendar.handleDateTimes import DateTimeHandler

class EventRequest(BaseModel):
    event_details: EventDetails
    target_start: str | None = None
    user_id: str


class TaskListModel:
    @staticmethod
    def fetch_events(event_request: EventRequest):
        """Fetch events for a user.

        Args:
            event_request (EventRequest): The request object containing event details and user ID.

        Returns:
            list[dict]: A list of events for the user.
        """
        event_details = event_request.event_details
        user_id = event_request.user_id
        request_setup = RequestSetup(event_details, user_id)
        events =  request_setup.calendar_insights.scheduled_events
        for event in events:
            yield event.model_dump()

    @staticmethod
    async def list_events(event_request: EventRequest) -> list[dict]:
        """List events for a user.

        Args:
            event_request (EventRequest): The request object containing event details and user ID.

        Returns:
            list[dict]: A list of formatted events for the user.
        """
        events = TaskListModel.fetch_events(event_request)
        target_start = event_request.target_start
        datetime_handler = DateTimeHandler(input_text="None")
        processed_events = []
        for event in events:
            if datetime_handler.verify_event_time(event["start"], target_start=target_start):
                formatted_event = datetime_handler.format_datetimes(event["start"], event["end"])
                event["start"] = formatted_event["start_time"]
                event["end"] = formatted_event["end_time"]
                processed_events.append(event)
        print(f"Processed Events: {processed_events}")
        return processed_events

    
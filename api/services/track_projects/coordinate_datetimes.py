from api.services.calendar.handleEvents import RequestSetup
from api.services.calendar.handleDateTimes import DateTimeSet
from api.schemas.model import EventOutput
from api.schemas.calendar import CalendarEvent, CalendarInsights
from datetime import datetime
from api.validation.handleProjects import ValidateProjectHandler

validator = ValidateProjectHandler()
class CoordinateDateTimes(RequestSetup):
    def __init__(self, user_id: str, request_start: datetime, request_end: datetime):
        self.request_start = request_start
        self.request_end = request_end
        calendar_event = CalendarEvent(
            datetime_obj = DateTimeSet(
                target_datetimes=[
                    (self.request_start, self.request_end)
                ]
            )
        )
        super().__init__(
            calendar_event=calendar_event,
            event_output=EventOutput(),
            user_id=user_id,
            calendar_service=None
        )
        self.fetch_events_list()
        self.user_events = self.calendar_insights.scheduled_events
        self.available = True

    @validator.validate_coordinator
    def coordinate(self) -> bool:
        """Coordinate the start and end times of events.

        Returns:
            bool: Whether the events are available.
        """
        for requested_start, requested_end in self.calendar_event.datetime_obj.target_datetimes:
            for event in self.user_events:
                if event:
                    event_start = event.start
                    event_end = event.end 

                if (requested_end > event_start and requested_end < event_end) or \
                (requested_start > event_start and requested_start < event_end):
                    self.available = False
                    return self.available
                

        return self.available
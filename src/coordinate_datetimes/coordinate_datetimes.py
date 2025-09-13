from src.google_calendar.handleEvents import RequestSetup
from src.google_calendar.handleDateTimes import DateTimeSet
from src.model_setup.structure_model_output import EventDetails
from datetime import datetime
from src.validators.validators import ValidateCoordinator

validator = ValidateCoordinator()
class CoordinateDateTimes(RequestSetup):
    def __init__(self, user_id: str, request_start: datetime, request_end: datetime):
        self.request_start = request_start
        self.request_end = request_end
        event_details = EventDetails(
            datetime_obj = DateTimeSet(
                target_datetimes=[
                    (self.request_start, self.request_end)
                ]
            )
        )
        super().__init__(event_details, user_id)
        self.user_events = self.calendar_insights.scheduled_events
        self.available = False

    @validator.validate_coordinator
    def coordinate(self) -> bool:
        """Coordinate the start and end times of events.

        Returns:
            bool: Whether the events are available.
        """
        for start_time, end_time in self.event_details.datetime_obj.target_datetimes:
            for event in self.user_events:
                event_start = event.start
                event_end = event.end if event.end else event.start

                if (start_time < event_start and end_time < event_end) or \
                (start_time > event_end):
                    self.available = False
                    return self.available
                

        return self.available
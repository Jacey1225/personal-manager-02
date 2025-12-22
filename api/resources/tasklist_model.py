from api.services.calendar.eventSetup import RequestSetup
from api.config.uniformInterface import UniformInterface
from api.schemas.calendar import EventsRequest
from api.schemas.model import EventOutput
from api.config.cache import async_cached, event_cache

class TaskListModel:
    @staticmethod
    @async_cached(event_cache)
    async def list_events(event_request: EventsRequest) -> list[dict]:
        """List events for a user.

        Args:
            event_request (EventRequest): The request object containing event details and user ID.

        Returns:
            list[dict]: A list of formatted events for the user.
        """
        calendar_event = event_request.calendar_event
        user_id = event_request.user_id
        minTime = event_request.minTime
        maxTime = event_request.maxTime
        service = await UniformInterface(user_id).fetch_service()

        if minTime and maxTime:
            request_setup = RequestSetup(
                calendar_event, 
                EventOutput(),
                user_id, 
                service,
                minTime=minTime, 
                maxTime=maxTime)
        else:
            request_setup = RequestSetup(
                calendar_event, 
                EventOutput(),
                user_id,
                service)
        request_setup.fetch_events_list()
        request_setup.calendar_insights.scheduled_events = \
        request_setup.datetime_handler.sort_datetimes(
            request_setup.calendar_insights.scheduled_events)

        events_dict = []
        for event in request_setup.calendar_insights.scheduled_events:
            if not event:
                print("Empty event encountered, skipping.")
                continue
            if event.start and event.end:
                event.start, event.end = request_setup.datetime_handler.format_datetimes(event.start, event.end)
            events_dict.append(event.model_dump())
        return events_dict
 
    
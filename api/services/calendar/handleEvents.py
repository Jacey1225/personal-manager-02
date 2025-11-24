from api.config.uniformInterface import UniformInterface
from api.services.model_setup.structure_model_output import EventOutput
from api.services.calendar.eventSetup import RequestSetup, CalendarInsights, CalendarEvent 
from api.validation.handleEventSetup import ValidateEventHandling
from api.config.plugins.enable_google_api import SyncGoogleEvents, SyncGoogleTasks
from api.config.plugins.enable_apple_api import SyncAppleEvents
from datetime import datetime, timezone
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

validator = ValidateEventHandling()

#MARK: add
class AddToCalendar(RequestSetup):
    def __init__(self, 
                 calendar_event: CalendarEvent, 
                 event_output: EventOutput,
                 user_id: str, 
                 calendar_service):
        super().__init__(
            calendar_event, 
            event_output, 
            user_id, 
            calendar_service)

    @validator.validate_request_status
    def add_event(self):
        """Add a new event to Google Calendar or a task to Google Tasks.

        Raises:
            ValueError: If event template is not initialized.
            ValueError: If no datetime objects are provided.
            RuntimeError: If an error occurs while adding the event/task.

        Returns:
            str: A message indicating the result of the operation.
        """
        try:
            for start_time, end_time in self.calendar_event.datetime_obj.target_datetimes:
                self.calendar_event.start = start_time
                self.calendar_event.end = end_time
                self.fetch_event_template() 
                try:
                    if not self.calendar_service:
                        raise ValueError("Calendar service is not initialized.")
                    if self.calendar_insights.template:
                        self.calendar_service.create_event(self.calendar_insights.template)
                except Exception as e:
                    print(f"An error occurred while creating the event within {self.add_event.__name__}: {e}")
                    return {"status": "error", "message": "Your calendar may not support this request."}

            return {"status": "success"}
        except Exception as e:
            print(f"An error occurred while adding the event/task: {e}")
            return {"status": "error", "message": str(e)}

#MARK: delete
class DeleteFromCalendar(RequestSetup):
    def __init__(self, 
                 calendar_event: CalendarEvent, 
                 event_output: EventOutput,
                 user_id: str, 
                 calendar_service):
        super().__init__(
            calendar_event, 
            event_output, 
            user_id, 
            calendar_service)

    @validator.validate_request_status
    def delete_event(self, event_id: str):
        """Delete an event from Google Calendar or a task from Google Tasks.

        Raises:
            ValueError: If no event is found to delete.
            RuntimeError: If an error occurs while deleting the event/task.

        Returns:
            str: A message indicating the result of the operation.
        """
        try:
            if event_id:
                calendar_event = next(
                    (event for event in self.calendar_insights.matching_events if \
                     event.event_id == event_id), 
                     None)
                if not calendar_event:
                    raise ValueError(f"No matching event found with ID '{event_id}'.")
                if not self.calendar_service:
                    raise ConnectionError("Calendar service is not initialized.")
                
                self.calendar_service.delete_event(event_id=event_id)
                return {"status": "success"}
            else:
                raise ValueError(f"No event found with ID '{event_id}'.")

        except Exception as e:
            print(f"An error occurred while deleting the event/task: {e}")
            return {"status": "error", "message": "We're having trouble processing this request at the moment"}

#MARK: Update
class UpdateFromCalendar(RequestSetup):
    def __init__(self,
                 calendar_event: CalendarEvent, 
                 event_output: EventOutput,
                 user_id: str, 
                 calendar_service):
        super().__init__(
            calendar_event, 
            event_output, 
            user_id, 
            calendar_service)
        
    @validator.log_target_elimination
    def eliminate_targets(self, event_id: str):

        """Eliminate target datetimes for a specific event ID to avoid irrelevant datetime instances

        Args:
            event_id (str): The ID of the event to eliminate targets for.
        """
        calendar_target = next(
            (event for event in self.calendar_insights.matching_events if \
             event.event_id == event_id), 
             None)

        all_target_datetimes = self.calendar_event.datetime_obj.target_datetimes.copy()
        for target_datetime in all_target_datetimes:
            start_target, end_target = target_datetime
            start_target = datetime.fromisoformat(start_target).replace(tzinfo=timezone.utc).astimezone(tz=None).isoformat()
            end_target = datetime.fromisoformat(end_target).replace(tzinfo=timezone.utc).astimezone(tz=None).isoformat() \
            if end_target else None
            
            if calendar_target:
                calendar_start = datetime.fromisoformat(calendar_target.start.isoformat()).replace(tzinfo=timezone.utc).astimezone(tz=None).isoformat()
                calendar_end = datetime.fromisoformat(calendar_target.end.isoformat()).replace(tzinfo=timezone.utc).astimezone(tz=None).isoformat() \
                if calendar_target.end else None

                if calendar_start and calendar_end:
                    if start_target == calendar_start and end_target == calendar_end:
                        print(f"Removing target datetime: {target_datetime} -> {calendar_target} == {start_target}, {end_target}")
                        self.calendar_event.datetime_obj.target_datetimes.remove(target_datetime)
                if calendar_start and not calendar_end:
                    if calendar_start == start_target:
                        print(f"Removing target datetime: {target_datetime} -> {calendar_target} == {start_target}")
                        self.calendar_event.datetime_obj.target_datetimes.remove(target_datetime)

    @validator.validate_request_status
    def update_event(self, event_id: str):
        """Update an existing event in Google Calendar or a task in Google Tasks.

        Raises:
            ValueError: If event name or personal email is not provided.
            RuntimeError: If an error occurs while updating the event/task.

        Returns:
            str: A message indicating the result of the operation.
        """
        if not self.calendar_event.datetime_obj.target_datetimes:
            raise ValueError("At least one datetime object must be provided to update an event.")
        
        try:
            if not self.calendar_service:
                raise ValueError("Calendar service is not initialized.")
            if event_id:
                for start_time, end_time in self.calendar_event.datetime_obj.target_datetimes:
                    original_event = self.calendar_service.fetch_event(
                        event_id=event_id)
                    if original_event:
                        original_event.start = start_time 
                        original_event.end = end_time
                        try:
                            self.calendar_service.update_event(original_event)
                        except Exception as e:
                            print(f"Error updating event: {e}")
                            return {"status": "error", "message": "Your calendar may not support this request"}
                    return {"status": "success"}
            else:
                raise ValueError(f"No event found with ID '{event_id}'.")

        except Exception as e:
            raise RuntimeError(f"An error occurred while updating the event/task: {e}")
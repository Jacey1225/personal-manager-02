from src.model_setup.structure_model_output import EventDetails
from src.google_calendar.eventSetup import RequestSetup, CalendarInsights
from src.validators.validators import ValidateEventHandling
from datetime import datetime, timezone
import pytz

validator = ValidateEventHandling()

#MARK: add
class AddToCalendar(RequestSetup):
    def __init__(self, event_details: EventDetails, user_id: str, personal_email: str = "jaceysimps@gmail.com"):
        super().__init__(event_details, user_id, personal_email)

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
                for start_time, end_time in self.event_details.datetime_obj.target_datetimes:
                    self.classify_request((start_time, end_time))
                    self.fetch_event_template()
                    if self.calendar_insights.is_event:
                        self.calendar_insights.template['start']['dateTime'] = start_time
                        self.calendar_insights.template['end']['dateTime'] = end_time
                        if start_time and end_time:
                            self.event_service.events().insert(calendarId=self.event_list_id, body=self.calendar_insights.template).execute() #type: ignore
                    else:
                        local_tz = pytz.timezone('America/Los_Angeles')
                        due_datetime = local_tz.localize(start_time)
                        self.calendar_insights.template['due'] = due_datetime.isoformat()
                        if due_datetime:
                            self.task_service.tasks().insert(tasklist=self.task_list_id, body=self.calendar_insights.template).execute() #type: ignore

                return {"status": "success"}
        except Exception as e:
            print(f"An error occurred while adding the event/task: {e}")
            return {"status": "error", "message": str(e)}

#MARK: delete
class DeleteFromCalendar(RequestSetup):
    def __init__(self, event_details: EventDetails, user_id: str, personal_email: str = "jaceysimps@gmail.com"):
        super().__init__(event_details, user_id, personal_email)

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
                calendar_event = next((event for event in self.calendar_insights.matching_events if event['event_id'] == event_id), None)
                if not calendar_event:
                    raise ValueError(f"No matching event found with ID '{event_id}'.")
                
                self.classify_request(event_id=event_id)
                if self.calendar_insights.is_event:
                    self.event_service.events().delete(calendarId=self.event_list_id, eventId=event_id).execute() #type: ignore
                else:
                    self.task_service.tasks().delete(tasklist=self.task_list_id, task=event_id).execute() #type: ignore
                return {"status": "success"}
            else:
                raise ValueError(f"No event found with ID '{event_id}'.")

        except Exception as e:
            raise RuntimeError(f"An error occurred while deleting the event/task: {e}")
        
#MARK: Update
class UpdateFromCalendar(RequestSetup):
    def __init__(self, event_details: EventDetails, user_id: str, personal_email: str = "jaceysimps@gmail.com"):
        super().__init__(event_details, user_id, personal_email)

    @validator.log_target_elimination
    def eliminate_targets(self, event_id: str):
        """Eliminate target datetimes for a specific event ID to avoid irrelevant datetime instances

        Args:
            event_id (str): The ID of the event to eliminate targets for.
        """
        calendar_target = next((event for event in self.calendar_insights.matching_events if event['event_id'] == event_id), None)

        all_target_datetimes = self.event_details.datetime_obj.target_datetimes.copy()
        for target_datetime in all_target_datetimes:
            start_target, end_target = target_datetime
            start_target = datetime.fromisoformat(start_target).replace(tzinfo=timezone.utc).astimezone(tz=None).isoformat()
            end_target = datetime.fromisoformat(end_target).replace(tzinfo=timezone.utc).astimezone(tz=None).isoformat() if end_target else None
            if calendar_target:
                calendar_start = datetime.fromisoformat(calendar_target['start']).replace(tzinfo=timezone.utc).astimezone(tz=None).isoformat()
                calendar_end = datetime.fromisoformat(calendar_target['end']).replace(tzinfo=timezone.utc).astimezone(tz=None).isoformat() if calendar_target['end'] else None
                if calendar_start and calendar_end:
                    if start_target == calendar_start and end_target == calendar_end:
                        print(f"Removing target datetime: {target_datetime} -> {calendar_target} == {start_target}, {end_target}")
                        self.event_details.datetime_obj.target_datetimes.remove(target_datetime)
                if calendar_start and not calendar_end:
                    if calendar_start == start_target:
                        print(f"Removing target datetime: {target_datetime} -> {calendar_target} == {start_target}")
                        self.event_details.datetime_obj.target_datetimes.remove(target_datetime)

    @validator.validate_request_status
    def update_event(self, event_id: str):
        """Update an existing event in Google Calendar or a task in Google Tasks.

        Raises:
            ValueError: If event name or personal email is not provided.
            RuntimeError: If an error occurs while updating the event/task.

        Returns:
            str: A message indicating the result of the operation.
        """
        if not self.event_details.datetime_obj.target_datetimes:
            raise ValueError("At least one datetime object must be provided to update an event.")
        
        try:
            if event_id:
                self.classify_request(event_id=event_id)
                if self.calendar_insights.is_event:
                    for start_time, end_time in self.event_details.datetime_obj.target_datetimes:
                        original_event = self.event_service.events().get(calendarId=self.event_list_id, eventId=event_id).execute() #type: ignore
                        if original_event['summary'] != self.event_details.event_name:
                            original_event['summary'] = self.event_details.event_name
                        original_event['start']['dateTime'] = start_time
                        original_event['end']['dateTime'] = end_time
                        self.event_service.events().update(calendarId=self.event_list_id, eventId=event_id, body=original_event).execute() #type: ignore
                else:
                    for due_time, _ in self.event_details.datetime_obj.target_datetimes:
                        original_task = self.task_service.tasks().get(tasklist=self.task_list_id, task=event_id).execute() #type: ignore
                        if original_task['title'] != self.event_details.event_name:
                            original_task['title'] = self.event_details.event_name
                        local_tz = pytz.timezone('America/Los_Angeles')
                        due_datetime = local_tz.localize(due_time)
                        original_task['due'] = due_datetime.isoformat()  # Use 'due' field for task due date/time
                    self.task_service.tasks().update(tasklist=self.task_list_id, task=event_id, body=original_task).execute() #type: ignore
                return {"status": "success"}
            else:
                raise ValueError(f"No event found with ID '{event_id}'.")

        except Exception as e:
            raise RuntimeError(f"An error occurred while updating the event/task: {e}")
import enable_api
from structure_model_output import EventDetails

class HandleEvents:
    def __init__(self, event_details: EventDetails, get_events: bool, personal_email: str):
        self.service = enable_api.enable_google_calendar_api()
        self.personal_email = personal_email
        self.event_details = event_details
        self.get_events = get_events
        self.task_list_id = '@default'
        if not self.service:
            raise ConnectionError("Failed to connect to Google Calendar API.")

        if self.event_details.event_name == "None" or self.event_details.event_date == "None" or self.event_details.event_time == "None":
            raise ValueError("Event name, date, and time must be provided to create an event.")
        self.event = {
            "summary": self.event_details.event_name,
            "start": {
                "dateTime": f"{self.event_details.event_date}T{self.event_details.event_time}:00",
                "timeZone": "America/Los_Angeles",
            },
            "attendees": [
                {"email": self.personal_email}
                ],
        }
        if self.get_events:
            self.events_list = self.service.tasks().list(tasklist=self.task_list_id).execute()  #type: ignore
            if self.events_list and 'items' in self.events_list:
                self.events = self.events_list['items']
            else:
                Warning("No events found.")


    def create_event(self):
        try:
            event_result = self.service.tasks().insert(tasklist=self.task_list_id, body=event, sendUpdates='all').execute() #type: ignore
            return event_result
        except Exception as e:
            raise RuntimeError(f"An error occurred while creating the event: {e}")
        
    def delete_event(self):
        try:
            for task in self.events:
                if task['title'] == self.event_details.event_name:
                    self.service.tasks().delete(tasklist=self.task_list_id, task=task['id']).execute()  #type: ignore
                    return f"Event '{self.event_details.event_name}' deleted successfully."
            return f"No event found with the name '{self.event_details.event_name}'."
        except Exception as e:
            raise RuntimeError(f"An error occurred while deleting the event: {e}")        
    
    def update_event(self):
        try:
            for task in self.events:
                if task['title'] == self.event_details.event_name:
                    task['due'] = f"{self.event_details.event_date}T{self.event_details.event_time}:00.000Z"
                    updated_task = self.service.tasks().update(tasklist=self.task_list_id, task=task['id'], body=task).execute()  #type: ignore
                    return updated_task
            return f"No event found with the name '{self.event_details.event_name}'."
        except Exception as e:
            raise RuntimeError(f"An error occurred while updating the event: {e}")

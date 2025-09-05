from google_calendar.enable_google_api import enable_google_calendar_api
from google_calendar.handleEvents import HandleEvents
from structure_model_output import EventDetails


service = enable_google_calendar_api()

if not service:
    raise RuntimeError("Failed to initialize Google Calendar service.")

class Project(HandleEvents):
    def __init__(self, project_name: str, event_details: EventDetails, personal_email):
        super().__init__(event_details, personal_email)
        self.service = service
        self.project_name = project_name
        self.project_items = []

    def __str__(self):
        return self.project_name

    def set_template_as_project(self):
        if self.template is None:
            raise ValueError("Event template is not initialized.")

        if self.is_event:
            self.template['description'] = f"Project: {self.project_name}"
        else:
            self.template['notes'] = f"Project: {self.project_name}"
    
    def fetch_project_events(self):
        if not self.project_name:
            raise ValueError("Project name is not set.")
        
        try:
            events_list = self.service.events().list(calendarId=self.event_id, q=self.project_name).execute() #type: ignore
            tasks_list = self.service.tasks().list(tasklist=self.task_list_id, q=self.project_name).execute() #type: ignore
            events_items = events_list.get('items', [])
            tasks_items = tasks_list.get('items', [])
            self.project_items = events_items + tasks_items
            if not self.project_items:
                return f"No events or tasks found for project '{self.project_name}'."
            return self.project_items
        except Exception as e:
            raise RuntimeError(f"An error occurred while fetching events/tasks for project '{self.project_name}': {e}")
    
    def delete_project(self):
        if not self.project_items:
            raise ValueError("No project items to delete. Fetch project events first.")
        
        for item in self.project_items:
            try:
                if 'description' in item:
                    self.service.events().delete(calendarId=self.event_id, eventId=item['id']).execute() #type: ignore
                elif 'notes' in item:
                    self.service.tasks().delete(tasklist=self.task_list_id, task=item['id']).execute() #type: ignore
            except Exception as e:
                print(f"Failed to delete item with ID {item['id']}: {e}")
        self.project_items = []
        return f"All events and tasks for project '{self.project_name}' have been deleted."
import google_calendar.enable_google_api as enable_google_api
from structure_model_output import EventDetails
from datetime import datetime, timedelta

def check_service(service):
    if service is None:
        raise ConnectionError("Failed to connect to Google Calendar API.")
    return True

class HandleEvents:
    def __init__(self, event_details: EventDetails, personal_email: str = "jaceysimps@gmail.com"):
        self.service = enable_google_api.enable_google_calendar_api()
        self.personal_email = personal_email
        self.event_details = event_details
        self.task_list_id = '@default'
        self.event_id = 'primary'

        if not self.event_details.event_name or not self.event_details.action:
            print(f"Event Name: {self.event_details.event_name}")
            print(f"Event Action: {self.event_details.action}")
            print(f"Input Text: {self.event_details.input_text}")
            raise ValueError("Event name and action must be provided in event_details.")
        if len(self.event_details.datetime_objs) < 1:
            raise ValueError("At least one datetime object must be provided in event_details.")
        
# Event Handling Setup
        self.all_scheduled = None
        self.is_event = False
        self.tasks_list = None
        self.events_list = None
        self.template = None
        self.fetch_events_list()
        self.verify_event()
        self.fetch_event_template()

    def verify_event(self):
        self.is_event = False
        if self.event_details.action.lower() == "add" and len(self.event_details.datetime_objs) > 1:
            self.is_event = True
        if self.all_scheduled:
            for scheduled in self.all_scheduled:
                if 'summary' in scheduled and scheduled['summary'] == self.event_details.event_name:
                    self.is_event = True
                    break
                if 'title' in scheduled and scheduled['title'] == self.event_details.event_name:
                    self.is_event = False
                    break
        return self.is_event
    
    def fetch_events_list(self):
        try:
            request_body = {
                'maxResults': 10,
                'singleEvents': True,
                'orderBy': 'startTime',
                'timeMin': datetime.now().isoformat() + 'Z', # 'Z' indicates UTC time
            }
            if check_service(self.service): #fetch existing events if needed
                self.tasks_list = self.service.tasks().list(tasklist=self.task_list_id, **request_body).execute() #type: ignore
                self.events_list = self.service.events().list(calendarId=self.event_id, **request_body).execute() #type: ignore
                self.all_scheduled = self.events_list.get('items', []) + self.tasks_list.get('items', []) #type: ignore
            else:
                raise ConnectionError("Google Calendar service is not available.")
        except AttributeError as e:
            print(f"Service attribute error: {e}")
            pass
        except Exception as e:
            raise RuntimeError(f"An error occurred while fetching events: {e}")
    
    def fetch_event_template(self):
        if not self.event_details.event_name or not self.personal_email:
            raise ValueError("Event name and personal email must be provided.")
        if not self.is_event:
            self.template = {
                'title': self.event_details.event_name,
                'notes': '',
                'due': None,
            }
        else:
            self.template = {
                'summary': self.event_details.event_name,
                'description': '',
                'start': {
                    'dateTime': None,
                    'timeZone': 'America/Los_Angeles',
                },
                'end': {
                    'dateTime': None,
                    'timeZone': 'America/Los_Angeles',
                },
                'attendees': [{'email': self.personal_email}],
            }
# --------------------------------------------------------------------- Constructor Methods --------------------------------------------------------------------- #
#MARK: Event Handling 
    
    def sort_due_dates(self): #ensure we get the right due date if multiple dates were found
        if not self.all_scheduled:
            raise ValueError("No scheduled events to sort.")
        
        start_obj = None
        end_obj = None
        due_obj = None
        target_due = None
        for event in self.all_scheduled:
            if 'summary' in event and event['summary'] == self.event_details.event_name:
                if 'start' in event and 'end' in event:
                    start = event['start'].get('dateTime', event['start'].get('date'))
                    end = event['end'].get('dateTime', event['end'].get('date'))
                    start_obj = datetime.fromisoformat(start.replace('Z', '+00:00')) if 'T' in start else datetime.fromisoformat(start)
                    end_obj = datetime.fromisoformat(end.replace('Z', '+00:00')) if 'T' in end else datetime.fromisoformat(end)
                    break
            if 'title' in event and event['title'] == self.event_details.event_name:
                if 'due' in event:
                    due = event['due']
                    due_obj = datetime.fromisoformat(due.replace('Z', '+00:00')) if 'T' in due else datetime.fromisoformat(due)
                    break
        if start_obj and end_obj:
            for dt in self.event_details.datetime_objs:
                if start_obj == dt or end_obj == dt:
                    continue
                else:
                    target_due = dt
                    break
        elif due_obj:
            for dt in self.event_details.datetime_objs:
                if due_obj == dt:
                    continue
                else:
                    target_due = dt
                    break
        if not target_due:
            raise ValueError("No valid due dates found after sorting.")
        
        return target_due, (end_obj - start_obj) if start_obj and end_obj else None
    
#MARK: add
    
    def add_event(self):
        if not self.template:
            raise ValueError("Event template is not initialized.")
        if not self.event_details.datetime_objs or len(self.event_details.datetime_objs) < 1:
            raise ValueError("At least one datetime object must be provided to add an event.")
        
        try:
            if self.is_event:
                target_start, duration = self.sort_due_dates()
                if not duration:
                    duration = timedelta(hours=1) #default to 1 hour if no duration found
                target_end = target_start + duration
                self.template['start']['dateTime'] = target_start.isoformat()
                self.template['end']['dateTime'] = target_end.isoformat()
                event = self.service.events().insert(calendarId=self.event_id, body=self.template).execute() #type: ignore
                return f"Event '{self.event_details.event_name}' added on {target_start.strftime('%Y-%m-%d %H:%M')}."
            else:
                target_due, _ = self.sort_due_dates()
                self.template['due'] = target_due.isoformat()
                task = self.service.tasks().insert(tasklist=self.task_list_id, body=self.template).execute() #type: ignore
                return f"Task '{self.event_details.event_name}' added with due date {target_due.strftime('%Y-%m-%d %H:%M')}."
        except Exception as e:
            raise RuntimeError(f"An error occurred while adding the event/task: {e}")
        
#MARK: delete
    
    def delete_event(self):
        if not self.template:
            raise ValueError("Event template is not initialized.")
        try:
            if self.is_event:
                events = self.service.events().list(calendarId=self.event_id, q=self.event_details.event_name).execute() #type: ignore
                for event in events.get('items', []):
                    if event['summary'] == self.event_details.event_name:
                        self.service.events().delete(calendarId=self.event_id, eventId=event['id']).execute() #type: ignore
                        return f"Event '{self.event_details.event_name}' deleted."
                return f"No matching event found for '{self.event_details.event_name}'."
            else:
                tasks = self.service.tasks().list(tasklist=self.task_list_id).execute() #type: ignore
                for task in tasks.get('items', []):
                    if task['title'] == self.event_details.event_name:
                        self.service.tasks().delete(tasklist=self.task_list_id, task=task['id']).execute() #type: ignore
                        return f"Task '{self.event_details.event_name}' deleted."
                return f"No matching task found for '{self.event_details.event_name}'."
        except Exception as e:
            raise RuntimeError(f"An error occurred while deleting the event/task: {e}")

#MARK: update
    
    def update_event(self):
        if not self.template:
            raise ValueError("Event template is not initialized.")
        if not self.event_details.datetime_objs or len(self.event_details.datetime_objs) < 1:
            raise ValueError("At least one datetime object must be provided to update an event.")
        
        try:
            if self.is_event:
                events = self.service.events().list(calendarId=self.event_id, q=self.event_details.event_name).execute() #type: ignore
                for event in events.get('items', []):
                    if event['summary'] == self.event_details.event_name:
                        target_start, duration = self.sort_due_dates()
                        if not duration:
                            duration = timedelta(hours=1) #default to 1 hour if no duration found
                        target_end = target_start + duration
                        event['start']['dateTime'] = target_start.isoformat()
                        event['end']['dateTime'] = target_end.isoformat()
                        updated_event = self.service.events().update(calendarId=self.event_id, eventId=event['id'], body=event).execute() #type: ignore
                        return f"Event '{self.event_details.event_name}' updated to {target_start.strftime('%Y-%m-%d %H:%M')}."
                return f"No matching event found for '{self.event_details.event_name}'."
            else:
                tasks = self.service.tasks().list(tasklist=self.task_list_id).execute() #type: ignore
                for task in tasks.get('items', []):
                    if task['title'] == self.event_details.event_name:
                        target_due, _ = self.sort_due_dates()
                        task['due'] = target_due.isoformat()
                        updated_task = self.service.tasks().update(tasklist=self.task_list_id, task=task['id'], body=task).execute() #type: ignore
                        return f"Task '{self.event_details.event_name}' updated with due date {target_due.strftime('%Y-%m-%d %H:%M')}."
                return f"No matching task found for '{self.event_details.event_name}'."
        except Exception as e:
            raise RuntimeError(f"An error occurred while updating the event/task: {e}")
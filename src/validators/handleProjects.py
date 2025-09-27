from typing import Callable
from datetime import datetime

class ValidateProjectHandler:
    @staticmethod
    def validate_user_data(func: Callable):
        def wrapper(self, *args, **kwargs):
            if not self.user_data:
                print(f"No user data found for host user: {self.user_id}")
                raise ValueError("User data not found.")
            
            project_id = kwargs.get("project_id")
            if not project_id:
                try:
                    project_id = args[0]
                except IndexError:
                    try:
                        project_id = self.project_id
                    except AttributeError:
                        print(f"No project ID found for host user: {self.user_id}")
                        raise ValueError("Project ID not found.")
            if not self.user_data.get("projects"):
                self.user_data["projects"] = {}

            if not self.user_data.get("projects_liked"):
                self.user_data["projects_liked"] = []

            return func(self, *args, **kwargs)
        return wrapper

    @staticmethod
    def validate_project_args(func: Callable):
        def wrapper(self, *args, **kwargs):
            for arg in args:
                print(f"Project argument: {arg}")
            result = func(self, *args, **kwargs)
            print(f"Project validation result: {result}")
            return result
        return wrapper

    @staticmethod
    def validate_coordinator(func: Callable):
        def wrapper(self):
            if not self.event_details.datetime_obj.target_datetimes:
                print(f"No target datetimes found: {self.request_start}, {self.request_end}")
                raise ValueError("Invalid event details.")
            if isinstance(self.request_start, str):
                if "T" in self.request_start:
                    self.request_start = datetime.fromisoformat(self.request_start)
                else:
                    print(f"Invalid start datetime format: {self.request_start}")
                    raise ValueError("Invalid start datetime format.")
            if isinstance(self.request_end, str):
                if "T" in self.request_end:
                    self.request_end = datetime.fromisoformat(self.request_end)
                else:
                    print(f"Invalid end datetime format: {self.request_end}")
                    raise ValueError("Invalid end datetime format.")

            if len(self.calendar_insights.scheduled_events) < 1:
                print(f"No events found")
                print(f"Calendar Insights: {self.calendar_insights.scheduled_events}")
                raise ValueError("No scheduled events found.")

            result = func(self)     
            return result
        return wrapper
    
    @staticmethod
    def validate_project_identifier(func: Callable):
        def wrapper(self, *args, **kwargs):
            if not self.user_data or not self.user_id:
                print(f"User data or host user is missing")
                raise ValueError("User data or host user is missing")

            if not self.user_data.get("projects"):
                print(f"No projects found for user: {self.user_id} before call {func.__name__}")
                result = self.event_details
            else:
                result = func(self, *args, **kwargs)

            print(f"Result: {result}")
            return result
        return wrapper
    
    @staticmethod
    def validate_project(func: Callable):
        def wrapper(self, *args, **kwargs):
            project_id = kwargs.get("project_id")
            if project_id is None and len(args) > 0:
                print(f"Project ID not found in kwargs, using args[0]: {args[0]}")
                project_id = args[0]
            
            print(f"Project ID: {project_id}")
            
            if project_id is None:
                raise ValueError("Unknown project ID.")
            
            if not hasattr(self, 'user_data') or not self.user_data:
                raise ValueError("User data not found.")
            
            if "projects" not in self.user_data:
                self.user_data["projects"] = {}
            
            if project_id not in self.user_data["projects"]:
                if func.__name__ in ['view_project', 'like_project', 'remove_like']:
                    pass
                else:
                    raise ValueError(f"Project ID {project_id} not found in user's projects.")
            
            result = func(self, *args, **kwargs)
            return result
        return wrapper
    
    @staticmethod
    def validate_project_events(func: Callable):
        def wrapper(self, *args, **kwargs):
            if not self.calendar_insights.scheduled_events:
                print(f"No scheduled events found.")
                raise ValueError("No scheduled events found.")
            
            result = func(self, *args, **kwargs)

            if not self.calendar_insights.project_events:
                print(f"No project events found after call {func.__name__}.")
                pass

            for i, project_event in enumerate(self.calendar_insights.project_events):
                event_start = project_event['start']
                event_end = project_event['end']
                if isinstance(event_start, datetime):
                    event_start = event_start.strftime("%A, %B %d, %Y %I:%M %p")
                    self.calendar_insights.project_events[i]['start'] = event_start
                elif isinstance(event_start, str):
                    event_start = datetime.strptime(event_start, "%A, %B %d, %Y %I:%M %p")
                    if not event_start:
                        print(f"Invalid start datetime format: {event_start}")
                        raise ValueError("Invalid start datetime format.")
                    self.calendar_insights.project_events[i]['start'] = event_start

                if isinstance(event_end, datetime):
                    event_end = event_end.strftime("%A, %B %d, %Y %I:%M %p")
                    self.calendar_insights.project_events[i]['end'] = event_end
                elif isinstance(event_end, str):
                    event_end = datetime.strptime(event_end, "%A, %B %d, %Y %I:%M %p")
                    if not event_end:
                        print(f"Invalid end datetime format: {event_end}")
                        raise ValueError("Invalid end datetime format.")
                    self.calendar_insights.project_events[i]['end'] = event_end

            print(f"Project Events: {self.calendar_insights.project_events}")
            return result
        return wrapper
    
    @staticmethod
    def validate_project_existence(func: Callable):
        def wrapper(self, *args, **kwargs):
            user_identifier = getattr(self, 'user_id', None) or getattr(self, 'user_id', None)
            
            if self.user_data:
                if self.user_data.get("projects"):
                    print(f"Projects found for user: {user_identifier} -> {self.user_data['projects']}")
                    return func(self, *args, **kwargs)

            print(f"No projects found for user: {user_identifier} -> {self.user_data}")
            return None
        return wrapper
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

            result = func(self, *args, **kwargs)
            print(f"Result from {func.__name__}: {result}")
            return result
        return wrapper

    @staticmethod
    def validate_project_args(func: Callable):
        def wrapper(self, *args, **kwargs):
            for arg in args:
                print(f"Project argument: {arg}")
            result = func(self, *args, **kwargs)
            print(f"Result from {func.__name__}: {result}")
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
            print(f"Result from {func.__name__}: {result}")  
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

            print(f"Result from {func.__name__}: {result}")
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
            print(f"Result from {func.__name__}: {result}")
            return result
        return wrapper
    
    @staticmethod
    def validate_project_events(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            print(f"Result from {func.__name__}: {result}")
            if isinstance(result, list):
                validated_events = []
                for event in result:
                    if isinstance(event, dict):
                        validated_event = event.copy()
                        
                        for time_field in ['start', 'end']:
                            if time_field in validated_event and validated_event[time_field] is not None:
                                time_value = validated_event[time_field]
                                if hasattr(time_value, 'strftime'):
                                    validated_event[time_field] = time_value.strftime("%A, %B %d, %Y %I:%M %p")
                        
                        validated_events.append(validated_event)
                    else:
                        validated_events.append(event)
                return validated_events
            
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
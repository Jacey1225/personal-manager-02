from typing import Callable

class ValidateEventHandling:
    @staticmethod
    def validate_events_list(func: Callable):
        def wrapper(self):
            result = func(self)

            if len(self.calendar_insights.scheduled_events) < 1:
                print(f"No events found")
                print(f"Calendar Insights: {self.calendar_insights.scheduled_events}")
                raise ValueError("No scheduled events found.")

            for event in self.calendar_insights.scheduled_events:
                if not event.event_name or not event.start or not event.event_id:
                    raise ValueError("Invalid event details found.")
                if event.event_name == '' or event.event_id == '':
                    raise ValueError("Empty event name or ID found.")
            print(f"Fetched {len(self.calendar_insights.scheduled_events)} events from calendar and tasks from {func.__name__}.")
            return result
        return wrapper

    @staticmethod
    def log_matching_events(func: Callable):
        def wrapper(self):
            result = func(self)
            print(f"Matching Events from {func.__name__}: {self.calendar_insights.matching_events}")
            return result
        return wrapper

    @staticmethod
    def validate_request_classifier(func: Callable):
        def wrapper(self, *args, **kwargs):
            num_null = 0
            for kwarg in kwargs:
                if kwarg is None:
                    num_null += 1

            if num_null > 1:
                raise ValueError("Too many null values found.")

            print(f"Matching Events: {self.calendar_insights.matching_events}")
            print(f"{func.__name__} called with args: {func.__annotations__}")
            result = func(self, *args, **kwargs)
            print(f"Request Classifier Result from {func.__name__}: {self.calendar_insights.is_event}")
            return result
        return wrapper
    
    @staticmethod
    def log_target_elimination(func: Callable):
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            print(f"{func.__name__} called with args: {func.__annotations__}")
            print(f"Remaining target datetimes from {func.__name__}: {self.event_details.datetime_obj.target_datetimes}")
            return result
        return wrapper

    @staticmethod
    def validate_request_status(func: Callable):
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            if result['status'] == 'error':
                print(f"Error occurred: {result['message']}")
                raise ValueError("An error occurred during request processing.")

            print(f"Request Status Result from {func.__name__}: {result['status']}")
            return result
        return wrapper


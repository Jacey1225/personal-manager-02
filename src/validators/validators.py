from typing import Callable
from datetime import datetime, time

class ValidateDateTimeSet:
    @staticmethod
    def validate_time_expressions(func: Callable):
        def wrapper(self):
            """Validates and fixes time expressions in the input text.

            Raises:
                ValueError: If the input types are invalid.

            Returns:
                Callable: The wrapped function with validated time expressions.
            """
            if not isinstance(self.input_text, str) or not isinstance(self.datetime_set.input_tokens, list):
                print(f"Invalid input types: {type(self.input_text)}, {type(self.datetime_set.input_tokens)}")
                raise ValueError("Invalid input types")
        
            input_text = self.input_text.strip()
            input_tokens = input_text.split(" ")
            for i, token in enumerate(input_tokens):
                if token[0].isdigit() and ("am" in token.lower() or "pm" in token.lower()):
                    print(f"Fixing token: {token}")
                    input_tokens[i] = token.lower().replace("am", " am").replace("pm", " pm")
                    print(f"Fixed token: {input_tokens[i]}")
            
            fixed_text = " ".join(input_tokens)
            self.datetime_set.input_tokens = fixed_text.split(" ")
            print(f"Input Tokens: {self.datetime_set.input_tokens}")

            result = func(self)

            if len(self.datetime_set.dates) == 0:
                print(f"Defaulting to today's date.")
                self.datetime_set.dates.append(datetime.now())
            if len(self.datetime_set.times) == 0:
                print(f"Defaulting to now.")
                self.datetime_set.times.append(time(datetime.now().hour, datetime.now().minute))

            print(f"Dates: {self.datetime_set.dates}, Times: {self.datetime_set.times}")
            return result
        return wrapper

    @staticmethod
    def validate_datetime_params(func: Callable):
        def wrapper(self) -> Callable:
            """Validates the datetime parameters.

            Raises:
                ValueError: Needs a corresponding time for each date.
                ValueError: Needs a corresponding date for each time.

            Returns:
                Callable: The wrapped function with validated datetime parameters.
            """
            if len(self.datetime_set.dates) < len(self.datetime_set.times) and \
            len(self.datetime_set.times) % len(self.datetime_set.dates) != 0:
                raise ValueError("The number of times must be a multiple of the number of dates to imply dates correctly.")
            if len(self.datetime_set.dates) > len(self.datetime_set.times) and \
            len(self.datetime_set.dates) % len(self.datetime_set.times) != 0:
                raise ValueError("The number of dates must be a multiple of the number of times to imply times correctly.")

            result = func(self)

            print(f"Organized datetimes: {self.datetime_set.datetimes}")
            return result
        return wrapper
    
    @staticmethod
    def log_target_datetimes(func: Callable):
        def wrapper(self):
            result = func(self)
            print(f"Target Datetimes: {self.datetime_set.target_datetimes}")
            return result
        return wrapper

class ValidateModelOutput:
    @staticmethod
    def validate_event_details(func: Callable):
        def wrapper(self):
            """Validates the event details.

            Raises:
                ValueError: If the event details are invalid.

            Returns:
                Callable: The wrapped function with validated event details.
            """
            result = func(self)

            if len(self.event_details.datetime_obj.target_datetimes) < 1:
                print(f"Event Input: {self.event_details.input_text}")
                raise ValueError("No valid target datetimes found.")
            if len(self.event_details.datetime_obj.dates) < 1:
                print(f"Event Input: {self.event_details.input_text}")
                raise ValueError("No valid target dates found.")
            if len(self.event_details.datetime_obj.times) < 1:
                print(f"Event Input: {self.event_details.input_text}")
                raise ValueError("No valid target times found.")

            if not self.event_details.action or len(self.event_details.event_name) < 1 or len(self.event_details.response) < 1:
                print(f"Model Response: {self.event_details.raw_output}")
                raise ValueError("Invalid event details.")

            print(f"Validated EventDetails: {self.event_details}")
            return result
        return wrapper
    
    @staticmethod
    def validate_response_process(func: Callable):
        def wrapper(self):
            if "." not in self.input_text and "?" not in self.input_text and "!" not in self.input_text:
                print(f"Cannot find a separator, defaulting to a single sentence.")
                self.input_text += "."

            result = func(self)

            if not result or len(result) < 1:
                print(f"Input Text: {self.input_text}")
                raise ValueError("No events were processed.")
            return result
        return wrapper
    
    @staticmethod
    def validate_text_to_speech(func: Callable):
        def wrapper(self, *args):
            for arg in args:
                if (isinstance(arg, str) and arg == "None") or not arg:
                    raise ValueError("No valid text provided for text-to-speech conversion.")
            return func(self, *args)
        return wrapper

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
            print(f"Fetched {len(self.calendar_insights.scheduled_events)} events from calendar and tasks.")
            return result
        return wrapper

    @staticmethod
    def log_matching_events(func: Callable):
        def wrapper(self):
            result = func(self)
            print(f"Matching Events: {self.calendar_insights.matching_events}")
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
            result = func(self, *args, **kwargs)
            print(f"Request Classifier Result: {self.calendar_insights.is_event}")
            return result
        return wrapper
    
    @staticmethod
    def log_target_elimination(func: Callable):
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            print(f"Remaining target datetimes: {self.event_details.datetime_obj.target_datetimes}")
            return result
        return wrapper

    @staticmethod
    def validate_request_status(func: Callable):
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            if result['status'] == 'error':
                print(f"Error occurred: {result['message']}")
                raise ValueError("An error occurred during request processing.")

            print(f"Request Status Result: {result['status']}")
            return result
        return wrapper

class ValidateProjectHandler:
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
            if not self.user_data or not self.host_user:
                print(f"User data or host user is missing")
                raise ValueError("User data or host user is missing")

            if not self.user_data.get("projects"):
                print(f"No projects found for user: {self.host_user} before call {func.__name__}")
                result = self.event_details
            else:
                result = func(self, *args, **kwargs)

            print(f"Result: {result}")
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
            if self.user_data:
                if self.user_data["projects"]:
                    print(f"Projects found for user: {self.username} -> {self.user_data['projects']}")
                    return func(self, *args, **kwargs)

            print(f"No projects found for user: {self.username} -> {self.user_data}")
            return None
        return wrapper
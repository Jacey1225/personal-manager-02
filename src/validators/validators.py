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
                raise ValueError("Invalid input types")
        
            input_text = self.input_text
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
                self.datetime_set.times.append(datetime.now().time())

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

            return func(self)
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
            if len(self.event_details.datetime_obj.target_datetimes) < 1:
                print(f"Event Input: {self.event_details.input_text}")
                raise ValueError("No valid target datetimes found.")
            if len(self.event_details.datetime_obj.dates) < 1:
                print(f"Event Input: {self.event_details.input_text}")
                raise ValueError("No valid target dates found.")
            if len(self.event_details.datetime_obj.times) < 1:
                print(f"Event Input: {self.event_details.input_text}")
                raise ValueError("No valid target times found.")
            
            result = func(self)

            if not self.event_details.action or len(self.event_details.event_name) < 1 or len(self.event_details.response) < 1:
                print(f"Model Response: {self.event_details.raw_output}")
                raise ValueError("Invalid event details.")

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
                print(f"Calendar Insights: {self.calendar_insights.scheduled_events}")
                raise ValueError("No scheduled events found.")

            for event in self.calendar_insights.scheduled_events:
                if not event.event_name or not event.start or not event.event_id:
                    raise ValueError("Invalid event details found.")
                if event.event_name == '' or event.event_id == '':
                    raise ValueError("Empty event name or ID found.")
                
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
            
            return func(self, *args, **kwargs)
        return wrapper
    
    @staticmethod
    def validate_request_status(func: Callable):
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            if result['status'] == 'error':
                print(f"Error occurred: {result['message']}")
                raise ValueError("An error occurred during request processing.")
            return result
        return wrapper

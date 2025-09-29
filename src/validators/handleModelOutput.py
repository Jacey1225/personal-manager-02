from typing import Callable

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
                raise ValueError(f"{func.__name__}, {func.__class__}: No valid target datetimes found.")
            if len(self.event_details.datetime_obj.dates) < 1:
                print(f"Event Input: {self.event_details.input_text}")
                raise ValueError(f"{func.__name__}, {func.__class__}: No valid target dates found.")
            if len(self.event_details.datetime_obj.times) < 1:
                print(f"Event Input: {self.event_details.input_text}")
                raise ValueError(f"{func.__name__}, {func.__class__}: No valid target times found.")

            if not self.event_details.action or len(self.event_details.event_name) < 1 or len(self.event_details.response) < 1:
                print(f"Model Response: {self.event_details.raw_output}")
                raise ValueError(f"{func.__name__}, {func.__class__}: Invalid event details.")

            print(f"Validated EventDetails from {func.__name__}, {func.__class__}: {self.event_details}")
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
                raise ValueError(f"{func.__name__}, {func.__class__}: No events were processed.")

            print(f"Result from {func.__name__}, {func.__class__}: {result}")
            return result
        return wrapper
    
    @staticmethod
    def validate_text_to_speech(func: Callable):
        def wrapper(self, *args):
            for arg in args:
                if (isinstance(arg, str) and arg == "None") or not arg:
                    raise ValueError(f"{func.__name__}, {func.__class__}: No valid text provided for text-to-speech conversion.")
            return func(self, *args)
        return wrapper

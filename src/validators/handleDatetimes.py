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

            print(f"Organized datetimes from {func.__name__}: {self.datetime_set.datetimes}")
            return result
        return wrapper
    
    @staticmethod
    def log_target_datetimes(func: Callable):
        def wrapper(self):
            result = func(self)
            print(f"Target Datetimes from {func.__name__}: {self.datetime_set.target_datetimes}")
            return result
        return wrapper
    
    @staticmethod
    def validate_time_verification(func: Callable):
        """Validates the time verification for the event.

        Args:
            func (Callable): The function to validate.
        """
        def wrapper(self, *args, **kwargs):
            for arg in args:
                if isinstance(arg, str):
                    try:
                        datetime.fromisoformat(arg)
                    except ValueError:
                        raise ValueError(f"Invalid datetime string: {arg} from {func.__name__}")
                    
            for kwarg in kwargs:
                if isinstance(kwarg, str):
                    try:
                        datetime.fromisoformat(kwarg)
                    except ValueError:
                        raise ValueError(f"Invalid datetime string: {kwarg} from {func.__name__}")
            
            try:
                result = func(self, *args, **kwargs)
            except Exception as e:
                print(f"Error verifying the time of event from {func.__name__}: {e}")
                result = None
            return result
        return wrapper
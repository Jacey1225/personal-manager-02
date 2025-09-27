from datetime import datetime, timedelta, time
from pydantic import BaseModel, Field
from typing import Union, Optional
from src.validators.handleDatetimes import ValidateDateTimeSet

validator = ValidateDateTimeSet()

class DateTimeSet(BaseModel):
    input_tokens: list[str] = Field(default=[], description="A list of all the input tokens found within an input text")
    times: list[time] = Field(default=[], description="A list of all the times found within an input text")
    dates: list[datetime] = Field(default=[], description="A list of all the dates found within an input text")
    datetimes: list[datetime] = Field(default=[], description="A list of all the datetime objects found within an input text")
    target_datetimes: list[tuple] = Field(default=[], description="A list of all the target datetime objects found within an input text as tuples representing start and end or due and None")


def fetch_days_ahead(target_day: int, current_day=datetime.now()):
    current_week_day = current_day.weekday()
    days_ahead = (target_day - current_week_day) % 7
    if days_ahead == 0:
        days_ahead = 7

    return current_day + timedelta(days=days_ahead)

def fetch_year_month(current_month: int, target_month: int):
    if target_month < current_month:
        return datetime(datetime.now().year + 1, target_month, datetime.now().day).strftime('%Y-%m-%d')
    return datetime(datetime.now().year, target_month, datetime.now().day).strftime('%Y-%m-%d') 

DATE_KEYS = {
    "today": datetime.now().strftime('%Y-%m-%d'),
    "tonight": datetime.now().strftime('%Y-%m-%d'),
    "evening": datetime.now().strftime('%Y-%m-%d'),
    "morning": datetime.now().strftime('%Y-%m-%d'),
    "afternoon": datetime.now().strftime('%Y-%m-%d'),
    "tomorrow": (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
    "monday": fetch_days_ahead(0).strftime('%Y-%m-%d'),
    "tuesday": fetch_days_ahead(1).strftime('%Y-%m-%d'),
    "wednesday": fetch_days_ahead(2).strftime('%Y-%m-%d'),
    "thursday": fetch_days_ahead(3).strftime('%Y-%m-%d'),
    "friday": fetch_days_ahead(4).strftime('%Y-%m-%d'),
    "saturday": fetch_days_ahead(5).strftime('%Y-%m-%d'),
    "sunday": fetch_days_ahead(6).strftime('%Y-%m-%d'),
    "january": fetch_year_month(datetime.now().month, 1),
    "february": fetch_year_month(datetime.now().month, 2),
    "march": fetch_year_month(datetime.now().month, 3),
    "april": fetch_year_month(datetime.now().month, 4),
    "may": fetch_year_month(datetime.now().month, 5),
    "june": fetch_year_month(datetime.now().month, 6),
    "july": fetch_year_month(datetime.now().month, 7),
    "august": fetch_year_month(datetime.now().month, 8),
    "september": fetch_year_month(datetime.now().month, 9),
    "october": fetch_year_month(datetime.now().month, 10),
    "november": fetch_year_month(datetime.now().month, 11),
    "december": fetch_year_month(datetime.now().month, 12)
}

class DateTimeHandler:
    def __init__(self, input_text, date_keys=DATE_KEYS):
        self.input_text = input_text.strip("?.!")
        self.date_keys = date_keys

        self.datetime_set = DateTimeSet(
            input_tokens=[],
            times=[],
            dates=[],
            datetimes=[],
            target_datetimes=[],
        )

    @validator.validate_time_expressions
    def compile_datetimes(self):
        """Compiles the input text into a set of datetime objects.
        """
        for i, token in enumerate(self.datetime_set.input_tokens):
            token = token.lower()
            if token in self.date_keys:
                if i < len(self.datetime_set.input_tokens) - 1:
                    if self.datetime_set.input_tokens[i+1][0].isdigit():
                        original_date = datetime.strptime(self.date_keys[token], '%Y-%m-%d')
                        target_day = int(''.join(filter(str.isdigit, self.datetime_set.input_tokens[i+1])))
                        original_date = original_date.replace(day=target_day)
                        self.datetime_set.dates.append(original_date)
                        continue
                parsed_date = datetime.strptime(self.date_keys[token], '%Y-%m-%d')
                self.datetime_set.dates.append(parsed_date)
            elif i < len(self.datetime_set.input_tokens) - 1 and token[0].isdigit() and self.datetime_set.input_tokens[i+1].lower() in ["am", "pm"]:
                if ":" in token:
                    hour = int(token[0:token.index(":")])
                    minute = int(token[token.index(":")+1:token.index(":")+3])
                else:
                    hour = int(token)
                    minute = 0

                if hour == 12 and "am" in self.datetime_set.input_tokens[i+1].lower():
                    hour = 0
                if "12" not in token and "pm" in self.datetime_set.input_tokens[i+1].lower():
                    hour = int(hour) + 12
                self.datetime_set.times.append(time(int(hour), int(minute)))
            else:
                continue

    @validator.validate_datetime_params
    def organize_for_datetimes(self):
        """Organizes dates and times into datetime objects within the datetime_set. The problem here is that we need to ensure that each date has a corresponding time.

        Raises:
            ValueError: The number of times is not a multiple of the number of dates when implying dates,
                        or if the number of dates is not a multiple of the number of times when implying times.

        Returns:
            DateTimeSet: The updated datetime_set with the combined datetime objects.
        """

        imply_dates = False
        imply_times = False

        if len(self.datetime_set.dates) < len(self.datetime_set.times):
            imply_dates = True
        elif len(self.datetime_set.dates) > len(self.datetime_set.times):
            imply_times = True
        else:
            for date_obj, time_obj in zip(self.datetime_set.dates, self.datetime_set.times):
                self.datetime_set.datetimes.append(datetime.strptime(f"{date_obj.date()} {time_obj}", '%Y-%m-%d %H:%M:%S'))
            return self.datetime_set

        if imply_dates:
            interval = len(self.datetime_set.times) // len(self.datetime_set.dates)
            for i in range(0, len(self.datetime_set.times), interval):
                for j in range(interval):
                    date_obj = self.datetime_set.dates[i // interval]
                    time_obj = self.datetime_set.times[i+j]
                    self.datetime_set.datetimes.append(datetime.strptime(f"{date_obj.date()} {time_obj}", '%Y-%m-%d %H:%M:%S'))

        if imply_times:
            interval = len(self.datetime_set.dates) // len(self.datetime_set.times)
            for i in range(0, len(self.datetime_set.dates), interval):
                for j in range(interval):
                    date_obj = self.datetime_set.dates[i+j]
                    time_obj = self.datetime_set.times[i // interval]
                    self.datetime_set.datetimes.append(datetime.strptime(f"{date_obj.date()} {time_obj}", '%Y-%m-%d %H:%M:%S'))
        return self.datetime_set
    
    @validator.log_target_datetimes
    def fetch_targets(self):
        """Fetch target datetimes for the event.
        """
        if len(self.datetime_set.datetimes) < 2:
            self.datetime_set.target_datetimes = [(dt, None) for dt in self.datetime_set.datetimes]
        else:
            for i in range(0, len(self.datetime_set.datetimes) - 1, 2):
                start_datetime = self.datetime_set.datetimes[i]
                end_datetime = self.datetime_set.datetimes[i + 1]

                if end_datetime:
                    if end_datetime > start_datetime:
                        self.datetime_set.target_datetimes.append((start_datetime, end_datetime))
                    else:
                        end_datetime = end_datetime.replace(day=end_datetime.day+1)
                        self.datetime_set.target_datetimes.append((start_datetime, end_datetime))
                else:
                    self.datetime_set.target_datetimes.append((start_datetime, None))
    
    def verify_event_time(self, event_start: Union[str, datetime]) -> bool:
        """Verify if the event start time is valid.

        Args:
            event_start (Union[str, datetime]): The start time of the event.

        Returns:
            bool: True if the event start time is valid, False otherwise.
        """
        current_date = datetime.now()
        if isinstance(event_start, str):
            event_start = datetime.fromisoformat(event_start)

        if event_start.date() < current_date.date():
            return False
                
        return True
    
    def sort_datetimes(self, scheduled_events: list) -> list:
        """Sorts the scheduled events by their start times.

        Args:
            scheduled_events (list): A list of scheduled events to sort.

        Returns:
            list: A list of sorted scheduled events.
        """
        sorted_events = scheduled_events.copy()
        count = 0
        while count < len(sorted_events):
            for i in range(len(sorted_events) - 1 - count):
                event_i_start = sorted_events[i].start if isinstance(sorted_events[i].start, datetime) else datetime.fromisoformat(sorted_events[i].start)
                event_next_start = sorted_events[i + 1].start if isinstance(sorted_events[i + 1].start, datetime) else datetime.fromisoformat(sorted_events[i + 1].start)
                if event_i_start > event_next_start:
                    sorted_events[i], sorted_events[i + 1] = sorted_events[i + 1], sorted_events[i]
            count += 1

        return sorted_events
    
    def format_datetimes(self, event_start: datetime, event_end: Optional[datetime]) -> dict:
        """Formats the start and end times of an event.

        Args:
            event_start (datetime): The start time of the event.
            event_end (Optional[datetime]): The end time of the event.

        Returns:
            dict: A dictionary containing the formatted start and end times.
        """
        start_formatted = event_start.strftime("%A, %B %d, %Y %I:%M %p")
        if event_end:
            end_formatted = event_end.strftime("%A, %B %d, %Y %I:%M %p")
        else:
            end_formatted = None
        return {
            "start_time": start_formatted,
            "end_time": end_formatted
        }

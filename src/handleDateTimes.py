from datetime import datetime, timedelta, time
from pydantic import BaseModel, Field
from typing import Optional

class DateTimeSet(BaseModel):
    input_tokens: list[str] = Field(default=[], description="A list of all the input tokens found within an input text")
    times: list[time] = Field(default=[], description="A list of all the times found within an input text")
    dates: list[datetime] = Field(default=[], description="A list of all the dates found within an input text")
    datetimes: list[datetime] = Field(default=[], description="A list of all the datetime objects found within an input text")
    target_datetimes: tuple = Field(default=(), description="A list of all the target datetime objects found within an input text as tuples representing start and end or due and None")
    is_event: Optional[bool] = Field(default=False, description="Indicates whether the extracted datetimes are part of an event")


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
            target_datetimes=(),
        )

    def compile_datetimes(self):
        """Compiles the input text into a set of datetime objects.
        """
        self.datetime_set.input_tokens = self.input_text.split(" ")
        for i, token in enumerate(self.datetime_set.input_tokens):
            token = token.lower()
            if token in self.date_keys:
                if i < len(self.datetime_set.input_tokens) and self.datetime_set.input_tokens[i+1][0].isdigit():
                    original_date = datetime.strptime(self.date_keys[token], '%Y-%m-%d')
                    target_day = int(''.join(filter(str.isdigit, self.datetime_set.input_tokens[i+1])))
                    original_date = original_date.replace(day=target_day)
                    self.datetime_set.dates.append(original_date)
                else:
                    parsed_date = datetime.strptime(self.date_keys[token], '%Y-%m-%d')
                    self.datetime_set.dates.append(parsed_date)
            if ":" in token:
                hour, minute = token.split(":")
                if "12:" in token and "am" in self.datetime_set.input_tokens[i+1].lower():
                    hour = 0
                if i < len(self.datetime_set.input_tokens) - 1 and "pm" in self.datetime_set.input_tokens[i+1].lower() and "12:" not in token:
                    hour = int(hour) + 12
                self.datetime_set.times.append(time(int(hour), int(minute)))
        

    def organize_for_datetimes(self):
        """Organizes dates and times into datetime objects within the datetime_set.

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
            if len(self.datetime_set.times) % len(self.datetime_set.dates) != 0:
                raise ValueError("The number of times must be a multiple of the number of dates to imply dates correctly.")
        elif len(self.datetime_set.dates) > len(self.datetime_set.times):
            imply_times = True
            if len(self.datetime_set.dates) % len(self.datetime_set.times) != 0:
                raise ValueError("The number of dates must be a multiple of the number of times to imply times correctly.")
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
                    if time_obj == time(0, 0, 0):
                        date_obj = date_obj.replace(day=date_obj.day+1)
                    self.datetime_set.datetimes.append(datetime.strptime(f"{date_obj.date()} {time_obj}", '%Y-%m-%d %H:%M:%S'))

        if imply_times:
            interval = len(self.datetime_set.dates) // len(self.datetime_set.times)
            for i in range(0, len(self.datetime_set.dates), interval):
                for j in range(interval):
                    date_obj = self.datetime_set.dates[i+j]
                    time_obj = self.datetime_set.times[i // interval]
                    self.datetime_set.datetimes.append(datetime.strptime(f"{date_obj.date()} {time_obj}", '%Y-%m-%d %H:%M:%S'))
        return self.datetime_set
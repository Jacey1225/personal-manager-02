from datetime import datetime, timedelta

class TimesDates:
    def __init__(self, input_text):
        self.input_text = input_text
        self.ambig_dates = [
            "today", "tomorrow", "yesterday", "now", "later", "soon",
            "next week", "last week", "next month", "last month",
            "next year", "last year", "afternoon", "evening", "morning",
            "midnight", "noon", "tonight", "this week", "this month",
            "this year", "next weekend", "last weekend", "next holiday",
        ]
        self.fixed_dates = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
            "PAD", "PAD", "PAD", "PAD", "PAD"
        ]
        

    def locate_dates(self):
        pass

    def locate_times(self):
        if ":" in self.input_text:
            time_str = self.input_text.split(":")[-1].strip()
        elif "am" in self.input_text.lower() or "pm" in self.input_text.lower():
            time_str = self.input_text.split()[-1].strip()

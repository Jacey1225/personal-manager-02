from transformers.models.t5 import T5Tokenizer, T5ForConditionalGeneration
from proxy_bypass import _configure_proxy_bypass
import os
import torch
import spacy
from datetime import datetime, timedelta
from pydantic import BaseModel

class EventDetails(BaseModel):
    event_name: str
    event_date: str
    event_time: str
    action: str
    response: str
    people: list[str]

class HandleResponse:
    def __init__(self, input_text, model_path='./model'):
        if not isinstance(input_text, str):
            print(f"Input type: {type(input_text)} --> {input_text}")
            raise TypeError("Input text must be a string.")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The model path '{model_path}' does not exist.")
        _configure_proxy_bypass() #bypass proxy for huggingface
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.eval()

        self.input_text = input_text
        self.nlp = spacy.load("en_core_web_sm")
        self.months = {
                'january': 1, 'february': 2, 'march': 3, 'april': 4,
                'may': 5, 'june': 6, 'july': 7, 'august': 8,
                'september': 9, 'october': 10, 'november': 11, 'december': 12
            }
        self.vague_dates = {
            "this afternoon": 0,"this morning": 0,"this evening": 0, "tonight": 0, 
            "today": 0, "tomorrow": 1, "day after tomorrow": 2,
            "next week": 7, "next month": 30, "next year": 365,
        }
        self.week_days = {
            "monday": 0, "tuesday": 1, "wednesday": 2,
            "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
        }
        
        self.event_details = EventDetails(
            event_name="None",
            event_date="None",
            event_time="None",
            action="None",
            response="None",
            people=[]
        )

    def fetch_base_entities(self): # get times, dates, people if any
        doc = self.nlp(self.input_text)
        for ent in doc.ents:
            time_entity = ent.text if ent.label_ == "TIME" else "None"
            date_entity = ent.text if ent.label_ == "DATE" else "None"
            person_entity = ent.text if ent.label_ == "PERSON" else "None"

        if time_entity == "None" and date_entity == "None" and person_entity == "None":
            print(f"Input Text: {self.input_text}")
            raise ValueError("No recognizable time, date, or person entities found in the input text.")
        return time_entity, date_entity, [person_entity]
    
    def format_date_time(self): #format the time and dates for Google Calendar modification request
        if self.event_details.event_date == "None" or self.event_details.event_time == "None":
            raise ValueError("Both event date and event time must be provided.")

        if "pm" in self.event_details.event_time.lower() and ":" in self.event_details.event_time:
            hour, minute = self.event_details.event_time.lower().replace("pm", "").strip().split(":")
            hour = int(hour) + 12 if int(hour) != 12 else 12
            self.event_details.event_time = f"{hour:02}:{minute}"
        elif "am" in self.event_details.event_time.lower() and ":" in self.event_details.event_time:
            hour, minute = self.event_details.event_time.lower().replace("am", "").strip().split(":")
            hour = int(hour) if int(hour) != 12 else 0
            self.event_details.event_time = f"{hour:02}:{minute}"
        else:
            print(f"Event Time: {self.event_details.event_time}")
            raise ValueError("Event time format is incorrect. Use 'HH:MM am/pm' format.")
        
        try:
            for month, month_num in self.months.items():
                if month in self.event_details.event_date.lower():
                    target_date = self.event_details.event_date.lower().replace(month, str(month_num))
                    self.event_details.event_date = datetime.strptime(target_date, '%m %d %Y').strftime('%Y-%m-%d')
                    return
            for vague, days_ahead in self.vague_dates.items():
                if vague in self.event_details.event_date.lower():
                    target_date = datetime.now() + timedelta(days=days_ahead)
                    self.event_details.event_date = target_date.strftime('%Y-%m-%d')
                    return
            for day, day_num in self.week_days.items():
                if day in self.event_details.event_date.lower():
                    today = datetime.now()
                    days_ahead = (day_num - today.weekday() + 7) % 7
                    days_ahead = days_ahead if days_ahead != 0 else 7
                    target_date = today + timedelta(days=days_ahead)
                    self.event_details.event_date = target_date.strftime('%Y-%m-%d')
                    return
        except Exception as e:
            print(f"Event Date: {self.event_details.event_date}")
            raise ValueError("Event date format is incorrect or unrecognized.") from e

    def generate_response(self): #fetch response from fine-tuned T5
        self.event_details.event_time, self.event_details.event_date, self.event_details.people = self.fetch_base_entities()
        if not isinstance(self.event_details.event_time, str) or not isinstance(self.event_details.event_date, str):
            print(f"Event Time: {self.event_details.event_time}, Event Date: {self.event_details.event_date}")
            raise TypeError("Extracted entities must be strings.")

        feature_context = f"\nEvent Time: {self.event_details.event_time}, \nEvent Date: {self.event_details.event_date}, " \
        f"\nEvent Input: {self.input_text}"
        inputs = self.tokenizer(feature_context, return_tensors="pt", padding=True, truncation=True)
        if inputs['input_ids'].shape[1] != 250 and inputs['attention_mask'].shape[1] != 250:
            print(inputs['input_ids'].shape, inputs['attention_mask'].shape)
            raise ValueError("Tokenized input length does not match expected length of 250.")
        
        with torch.no_grad():
            outputs = self.model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=250)
        
        if outputs[0].shape[1] != 250:
            print(outputs[0].shape)
            raise ValueError("Generated output length does not match expected length of 250.")
        
        self.event_details.response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def parse_response(self): #format response in a more accessible way
        if not isinstance(self.event_details.response, str):
            raise TypeError("Response must be a string.")
        try:
            self.event_details.event_name = self.event_details.response.split("Action: ")[0].split("Event: ")[1].strip().rstrip(',')
            self.event_details.action = self.event_details.response.split("Action: ")[1].split("Desired Response: ")[0].strip()
            self.event_details.response = self.event_details.response.split("Desired Response: ")[1].strip()
        except IndexError as e:
            print(f"Response: {self.event_details.response}")
            raise ValueError("Response format is incorrect. Unable to parse.") from e

        if self.event_details.event_name == '' or self.event_details.action == '' or self.event_details.response == '':
            print(f"Response: {self.event_details.response}")
            raise ValueError("Parsed values cannot be empty.")

    def process_response(self) -> list[EventDetails]:
        found_events = []
        if '.' in self.input_text:
            self.input_text = self.input_text.split('.')
        for text in self.input_text:
            self.generate_response()
            self.parse_response()
            if self.event_details.event_date != "None" and self.event_details.event_time != "None":
                self.format_date_time()
            found_events.append(self.event_details)
            self.event_details = EventDetails(
                event_name="None",
                event_date="None",
                event_time="None",
                action="None",
                response="None",
                people=[]
            )
        if not found_events:
            raise ValueError("No events were processed.")
        return found_events

from transformers.models.t5 import T5Tokenizer, T5ForConditionalGeneration
from proxy_bypass import _configure_proxy_bypass
import os
import torch
from pydantic import BaseModel
import datefinder
from datetime import datetime
from gtts import gTTS
import pygame
import io

class EventDetails(BaseModel):
    input_text: str 
    event_name: str
    datetime_objs: list = []
    action: str
    response: str

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
        
        self.event_details = EventDetails(
            input_text=self.input_text,
            event_name="None",
            datetime_objs=[],
            action="None",
            response="None",
        )

    def fetch_base_entities(self): # get times, dates, people if any
        found_dates = list(datefinder.find_dates(self.input_text))
        self.event_details.datetime_objs = found_dates
        return found_dates

    def generate_response(self): #fetch response from fine-tuned T5
        found_dates = self.fetch_base_entities()
        if not isinstance(self.event_details.datetime_objs, list):
            print(f"Event Date Type: {type(self.event_details.datetime_objs)} --> {self.event_details.datetime_objs}")
            raise TypeError("Extracted entities must be lists.")
        date_str = ''
        time_str = ''
        for dt in self.event_details.datetime_objs:
            if not isinstance(dt, datetime):
                print(f"Datetime Object Type: {type(dt)} --> {dt}")
                raise TypeError("All extracted datetime objects must be of type datetime.")
            date_str += dt.strftime("%B %d, %Y") + ' | '
            time_str += dt.strftime("%I:%M %p") + ' | '

        feature_context = f"\nEvent Time: {time_str}, \nEvent Date: {date_str}, " \
                          f"\nEvent Input: {self.input_text}"
        inputs = self.tokenizer(feature_context, return_tensors="pt", padding='max_length', truncation=True, max_length=250)
        if inputs['input_ids'].shape[1] != 250 and inputs['attention_mask'].shape[1] != 250:
            print(inputs['input_ids'].shape, inputs['attention_mask'].shape)
            raise ValueError("Tokenized input length does not match expected length of 250.")
        
        with torch.no_grad():
            outputs = self.model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=250)
        
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

    def process_response(self) -> list[EventDetails]: #process all found responses as EventDetails obj
        found_events = []
        if '.' in self.input_text:
            input_requests = self.input_text.split('.')
        elif '?' in self.input_text:
            input_requests = self.input_text.split('?')

        if not input_requests:
            print(f"Input Text: {self.input_text}")
            print(f"Input Requests: {input_requests}")
            raise ValueError("No valid input requests found.")
        for text in input_requests:
            if not text.strip():
                continue
            self.input_text = text.strip()
            self.generate_response()
            self.parse_response()
            found_events.append(self.event_details)
            self.event_details = EventDetails(
                input_text=self.input_text,
                event_name="None",
                datetime_objs=[],
                action="None",
                response="None",
            )
        if not found_events:
            raise ValueError("No events were processed.")
        return found_events
    
    def convert_response_to_speech(self, text): #enable the response to be spoken aloud for entertainment purposes
        if text == "None" or not text:
            raise ValueError("No response available to convert to speech.")
        tts = gTTS(text=text, lang='en')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)

        pygame.mixer.init()
        pygame.mixer.music.load(fp, 'mp3')
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue


if __name__ == "__main__":
    example_input = "Can you change my plans to do a bible study at 3pm to 4pm tomorrow."
    handler = HandleResponse(example_input)
    events = handler.process_response()
    for event in events:
        handler.convert_response_to_speech(event.response)
        print(f"Event_name: {event.event_name}\nEvent Dates: {event.datetime_objs}\nAction: {event.action}\nResponse: {event.response}\n")

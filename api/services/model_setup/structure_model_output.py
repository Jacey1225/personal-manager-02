from transformers import T5Tokenizer, T5ForConditionalGeneration
from api.validation.handleModelOutput import ValidateModelOutput
from api.services.google_calendar.handleDateTimes import DateTimeHandler
from api.schemas.model import EventDetails
from api.schemas.calendar import DateTimeSet
import os
import torch
from gtts import gTTS
import pygame
import io

validator = ValidateModelOutput()

class HandleResponse:
    def __init__(self, input_text, model_path='./model'):
        if not isinstance(input_text, str):
            print(f"Input type: {type(input_text)} --> {input_text}")
            raise TypeError("Input text must be a string.")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The model path '{model_path}' does not exist.")
        
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.eval()

        self.input_text = input_text
        
        self.event_details = EventDetails(
            input_text=self.input_text,
            event_name="None",
            datetime_obj=DateTimeSet(),
            action="None",
            response="None",
        )

        self.datetime_handler = DateTimeHandler(self.input_text)

    @validator.validate_event_details
    def generate_response(self): 
        """Generates a response from the model based on the input text.

        Raises:
            ValueError: If no valid dates or times are found in the input text.
            ValueError: If the response format is incorrect.
        """
        self.datetime_handler.compile_datetimes()
        self.datetime_handler.organize_for_datetimes()
        self.datetime_handler.fetch_targets()
        self.event_details.datetime_obj = self.datetime_handler.datetime_set
        
        date_str = ' | '.join([dt.strftime("%Y-%m-%d") for dt in self.datetime_handler.datetime_set.dates]) #type: ignore
        time_str = ' | '.join([dt.strftime("%I:%M %p") for dt in self.datetime_handler.datetime_set.times]) if len(self.datetime_handler.datetime_set.times) > 0 else "None"

        feature_context = f"\nEvent Time: {time_str}, \nEvent Date: {date_str}, " \
                          f"\nEvent Input: {self.input_text}"
        inputs = self.tokenizer(feature_context, return_tensors="pt", padding='max_length', truncation=True, max_length=250)
        if inputs['input_ids'].shape[1] != 250 and inputs['attention_mask'].shape[1] != 250:
            print(inputs['input_ids'].shape, inputs['attention_mask'].shape)
            raise ValueError("Tokenized input length does not match expected length of 250.")
        
        with torch.no_grad():
            outputs = self.model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=250)
        
        self.event_details.raw_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def parse_response(self): 
        """Parses the model's response into structured event details.

        Raises:
            TypeError: If the response is not a string.
            ValueError: If the response format is incorrect.
            ValueError: If any of the parsed values are empty.
        """
        try:
            self.event_details.event_name = self.event_details.raw_output.split("Action: ")[0].split("Event: ")[1].strip().rstrip(',')
            self.event_details.action = self.event_details.raw_output.split("Action: ")[1].split("Desired Response: ")[0].strip()
            self.event_details.response = self.event_details.raw_output.split("Desired Response: ")[1].strip()
        except IndexError as e:
            print(f"Response: {self.event_details.raw_output}")
            raise ValueError("Response format is incorrect. Unable to parse.") from e

    @validator.validate_response_process
    def process_response(self) -> list[EventDetails]: 
        """Processes the model's response into a list of EventDetails objects.

        Raises:
            ValueError: if the input text is empty
            ValueError: if the response format is incorrect

        Returns:
            list[EventDetails]: a list of EventDetails objects
        """
        found_events = []
        if '.' in self.input_text:
            input_requests = self.input_text.split('.')
        elif '?' in self.input_text:
            input_requests = self.input_text.split('?')

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
                action="None",
                response="None",
            )
        return found_events
    
    @validator.validate_text_to_speech
    def convert_response_to_speech(self, text): 
        """Converts the given text response to speech.

        Args:
            text (str): The text response to convert.

        Raises:
            ValueError: If no valid text is provided.
        """
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
    example_input = "Can you delete my work shift today?"
    handler = HandleResponse(example_input)
    events = handler.process_response()
    for event in events:
        handler.convert_response_to_speech(event.response)
        print(f"Event_name: {event.event_name}\n Dates: {handler.datetime_handler.datetime_set.dates}\nAction: {event.action}\nResponse: {event.response}\n")

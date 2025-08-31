from transformers.models.t5 import T5Tokenizer, T5ForConditionalGeneration
from proxy_bypass import _configure_proxy_bypass
import os
import torch
import spacy

class HandleResponse:
    def __init__(self, input_text, model_path='./model'):
        if not isinstance(input_text, str):
            raise TypeError("Input text must be a string.")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The model path '{model_path}' does not exist.")
        _configure_proxy_bypass() #bypass proxy for huggingface
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.eval()

        self.input_text = input_text
        self.nlp = spacy.load("en_core_web_sm")

    def fetch_base_entities(self): # get times, dates, people if any
        doc = self.nlp(self.input_text)
        for ent in doc.ents:
            time_entity = ent.text if ent.label_ == "TIME" else "None"
            date_entity = ent.text if ent.label_ == "DATE" else "None"
            person_entity = ent.text if ent.label_ == "PERSON" else "None"
        return time_entity, date_entity, person_entity

    def generate_response(self): #fetch response from fine-tuned T5
        time_entity, date_entity, person_entity = self.fetch_base_entities()
        if not isinstance(time_entity, str) or not isinstance(date_entity, str):
            raise TypeError("Extracted entities must be strings.")
    
        feature_context = f"\nEvent Time: {time_entity}, \nEvent Date: {date_entity}, " \
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
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def parse_response(self, response): #format response in a more accessible way
        if not isinstance(response, str):
            raise TypeError("Response must be a string.")
        try:
            event_classification = response.split("Action: ")[0].split("Event: ")[1].strip().rstrip(',')
            action_classification = response.split("Action: ")[1].split("Desired Response: ")[0].strip()
            desired_response = response.split("Desired Response: ")[1].strip()
        except IndexError as e:
            raise ValueError("Response format is incorrect. Unable to parse.") from e
        
        if event_classification == '' or action_classification == '' or desired_response == '':
            print(f"Response: {response}")
            raise ValueError("Parsed values cannot be empty.")
        
        return event_classification, action_classification, desired_response
    
    def handle(self):
        response = self.generate_response()
        event_classification, action_classification, desired_response = self.parse_response(response)
        # Further processing...

        return {
            "event_classification": event_classification,
            "action_classification": action_classification,
            "desired_response": desired_response
        }
from transformers import T5Tokenizer
import torch
import pandas as pd
import os
from pydantic import BaseModel 
from proxy_bypass import _configure_proxy_bypass

class EventRequest(BaseModel):
    input_text: str #user input
    response_text: str #expected response
    intent: str #user intent
    event_name: str #name of the event
    event_date: str #date of the event
    event_time: str #time of the event
    feature_context: str = '' #context for the model (intent + event details)
    feature_response: str = '' #context for the model (expected response)

class EventScheduler:
    def __init__(self, filename: str='event_scheduling', batch_size: int=8):
        _configure_proxy_bypass() #bypass proxy for huggingface
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.batch_size = batch_size

        if not os.path.exists('data'): #check dir existence
            os.makedirs('data')

        if not os.path.exists(f'data/{filename}.csv'): #check raw data path existence
            raise FileNotFoundError(f"The file 'data/{filename}.csv' does not exist.")

        if not os.path.exists(f'data/processed_event_data.pt'): #check processed data path existence
            processed_data = {
                "input_ids": torch.empty((0, 250), dtype=torch.long), #input_ids
                "attention_mask": torch.empty((0, 250), dtype=torch.long), #attention_mask
                "labels": torch.empty((0, 250), dtype=torch.long), #labels
            }
            torch.save(processed_data, f'data/processed_event_data.pt')

        self.data = pd.read_csv(f'data/{filename}.csv')
        columns = ['input_text', 'response_text', 'type', 'event_name', 'event_time', 'event_date']
        for col in columns: #check column data
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col} -- {self.data.columns.tolist()}")
        
        self.clean_data = self.data.dropna(subset=columns).reset_index(drop=True) #drop rows with missing values

    def fetch_feature(self):
        """Fetch features from the event data.

        Raises:
            TypeError: If any of the event fields are not strings.
            ValueError: If feature_context and feature_response are not provided.
        """
        for col, content in self.event:
            if not isinstance(content, str):
                raise TypeError(f"Field {content} is not of type str")

        feature_context = f"\nEvent Time: {self.event.event_time}," \
        f"\nEvent Date: {self.event.event_date}, " \
        f"\nEvent Input: {self.event.input_text}"

        feature_response = f"Event: {self.event.event_name}, Action: {self.event.intent}" \
        f"Desired Response: {self.event.response_text}"
        if self.event.feature_context == '' and self.event.feature_response == '':
            self.event.feature_context = feature_context
            self.event.feature_response = feature_response
        else:
            raise ValueError("feature_context and feature_response must be provided.")
            
    def tokenize_feature(self):
        """Tokenizes the feature context and response for model input.

        Raises:
            ValueError: If feature_context and feature_response are not set.

        Returns:
            Tuple[Dict[str, Tensor], Dict[str, Tensor]]: The tokenized context and response.
        """
        if self.event.feature_context == '' or self.event.feature_response == '':
            raise ValueError("feature_context and feature_response must be set before tokenization.")
        
        context_encoding = self.tokenizer(
            self.event.feature_context,
            max_length=250,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        response_encoding = self.tokenizer(
            self.event.feature_response,
            max_length=250,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return context_encoding, response_encoding

    def process_batch(self):
        """Processes a batch of event data.

        Raises:
            ValueError: If the batch data is empty.
            ValueError: If the tokenized sequences are not of the expected length.
        """
        if self.clean_data.empty:
            raise ValueError("No data available to process.")
        for i in range(0, len(self.clean_data), self.batch_size):
            batch_data = self.clean_data.iloc[i:i + self.batch_size]
            batch_events = {
                "input_ids": torch.empty((0, 250), dtype=torch.long), #input_ids
                "attention_mask": torch.empty((0, 250), dtype=torch.long), #attention_mask
                "labels": torch.empty((0, 250), dtype=torch.long), #labels
            }
            for _, row in batch_data.iterrows():
                self.event = EventRequest(
                    input_text=row['input_text'],
                    response_text=row['response_text'],
                    intent=row['type'],
                    event_name=row['event_name'],
                    event_time=row['event_time'],
                    event_date=row['event_date']
                )
                self.fetch_feature()
                context_encoding, response_encoding = self.tokenize_feature()
                if context_encoding['input_ids'].shape[1] != 250 or response_encoding['input_ids'].shape[1] != 250:
                    raise ValueError("Tokenized sequences must have a length of 250.")

                batch_events["input_ids"] = torch.cat((batch_events["input_ids"], context_encoding['input_ids']), dim=0)
                batch_events["attention_mask"] = torch.cat((batch_events["attention_mask"], context_encoding['attention_mask']), dim=0)
                batch_events["labels"] = torch.cat((batch_events["labels"], response_encoding['input_ids']), dim=0)
            
            processed_data = torch.load('data/processed_event_data.pt')
            processed_data["input_ids"] = torch.cat((processed_data["input_ids"], batch_events["input_ids"]), dim=0)
            processed_data["attention_mask"] = torch.cat((processed_data["attention_mask"], batch_events["attention_mask"]), dim=0)
            processed_data["labels"] = torch.cat((processed_data["labels"], batch_events["labels"]), dim=0)
            torch.save(processed_data, 'data/processed_event_data.pt')
            print(f"\rProcessed batch {i // self.batch_size + 1} out of {len(self.clean_data) // self.batch_size + 1}", end='', flush=True)

if __name__ == "__main__":
    scheduler = EventScheduler()
    scheduler.process_batch()
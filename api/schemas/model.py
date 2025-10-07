from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, time

class DateTimeSet(BaseModel):
    input_tokens: list[str] = Field(default=[], description="A list of all the input tokens found within an input text")
    times: list[time] = Field(default=[], description="A list of all the times found within an input text")
    dates: list[datetime] = Field(default=[], description="A list of all the dates found within an input text")
    datetimes: list[datetime] = Field(default=[], description="A list of all the datetime objects found within an input text")
    target_datetimes: list[tuple] = Field(default=[], description="A list of all the target datetime objects found within an input text as tuples representing start and end or due and None")

class EventDetails(BaseModel):
    input_text: str = Field(default="None", description="Raw input text from a user")
    raw_output: str = Field(default="None", description="The raw output from the model")
    event_name: str = Field(default="None", description="The identified name of the event inside the input text")
    datetime_obj: DateTimeSet = Field(default_factory=DateTimeSet, description="List of datetime objects extracted from the input text")
    action: str = Field(default="None", description="The action to be performed on the event (add, delete, update)")
    response: str = Field(default="None", description="The response generated for the event")
    transparency: str = Field(default="opaque", description="The transparency of the event (opaque, transparent, etc.)")
    guestsCanModify: bool = Field(default=False, description="Indicates if guests can modify the event")
    description: str = Field(default="None", description="The description of the event")
    attendees: List[str] = Field(default_factory=list, description="List of attendees for the event")

class InputRequest(BaseModel):
    input_text: str
    user_id: str
    
class EventRequest(BaseModel):
    input_text: str #user input
    response_text: str #expected response
    intent: str #user intent
    event_name: str #name of the event
    event_date: str #date of the event
    event_time: str #time of the event
    feature_context: str = '' #context for the model (intent + event details)
    feature_response: str = '' #context for the model (expected response)

class TrainingConfig(BaseModel):
    model_name: str = 't5-small'
    output_dir: str = './model'
    num_train_epochs: int = 20
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 5e-04
    weight_decay: float = 0.01
    logging_dir: str = './logs'
    logging_steps: int = 10
    evaluation_strategy: str = 'epoch'
    save_strategy: str = 'epoch'
    save_total_limit: int = 2
    predict_with_generate: bool = True
    fp16: bool = False
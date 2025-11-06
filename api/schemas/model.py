from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, time

class InputRequest(BaseModel):
    input_text: str
    user_id: str
    
class EventOutput(BaseModel):
    input_text: str = Field(default="", description="The raw input text from the user")
    raw_output: str = Field(default="", description="The raw output text from the model")
    response_text: str = Field(default="", description="The expected response text")
    intent: str = Field(default="", description="The intent of the user")
    feature_context: str = Field(default='', description="Context for the model (intent + event details)")
    feature_response: str = Field(default='', description="Context for the model (expected response)")

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
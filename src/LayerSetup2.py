import torch
import torch.nn as nn
from transformers.models.t5 import T5Config, T5ForConditionalGeneration

class EventLayers(nn.Module):
    def __init__(self, input_embedding_size, hidden_size, output_size=768):
        super().__init__()
        self.input_embedding_size = input_embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.event_loss = nn.MSELoss()
        self.intent_loss = nn.MSELoss()

        self.event_sequential = nn.Sequential(
            nn.Linear(self.input_embedding_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.output_size),
        )
        self.intent_sequential = nn.Sequential(
            nn.Linear(self.input_embedding_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.output_size),
        )
    
class T5Layers(nn.Module):
    def __init__(self, model_name='t5-base'):
        super().__init__()
        self.t5_config = T5Config.from_pretrained(model_name)
        self.t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
import torch
import torch.nn as nn
from transformers.models.t5 import T5Config, T5ForConditionalGeneration

class EventLayers(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.event_sequential = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
            nn.Sigmoid(),
        )

        self.intent_sequential = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        self.event_loss = nn.BCELoss()
        self.intent_loss = nn.MSELoss()

class T5Layers(nn.Module):
    def __init__(self, input_size, projection_size, t5_vocab_size, model_name='t5-base'):
        super().__init__()
        self.t5_projection = nn.Linear(input_size, projection_size) # 100 to 768
        self.embeddings_to_logits = nn.Linear(projection_size, t5_vocab_size) # 768 to 32128
        self.t5_config = T5Config.from_pretrained(model_name)
        self.t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
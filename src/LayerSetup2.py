import torch
import torch.nn as nn
from transformers.models.t5 import T5Config, T5ForConditionalGeneration

class EventLayers(nn.Module):
    def __init__(self, input_embedding_size, input_pos_size, hidden_size, output_size=768):
        super().__init__()
        self.input_embedding_size = input_embedding_size
        self.input_pos_size = input_pos_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.event_combined = nn.Sequential(
            nn.Linear(output_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
            nn.Sigmoid()
        )
        self.intent_combined = nn.Sequential(
            nn.Linear(output_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
            nn.Sigmoid()
        )

        self.event_loss = nn.MSELoss()
        self.intent_loss = nn.MSELoss()

    def get_embedding_sequential(self, input_embed):
        input_embed_sequential = nn.Sequential(
            nn.Linear(self.input_embedding_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.output_size),
            nn.Sigmoid()
        )
        return input_embed_sequential(input_embed)
    
    def get_pos_sequential(self, input_pos):
        input_pos_sequential = nn.Sequential(
            nn.Linear(self.input_pos_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.output_size),
            nn.Sigmoid()
        )

        return input_pos_sequential(input_pos)
    
class T5Layers(nn.Module):
    def __init__(self, input_size, projection_size, t5_vocab_size, model_name='t5-base'):
        super().__init__()
        self.embeddings_to_logits = nn.Linear(projection_size, t5_vocab_size) # 768 to 32128
        self.t5_config = T5Config.from_pretrained(model_name)
        self.t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
import torch
import torch.nn as nn
from transformers.models.t5 import T5Tokenizer, T5Model, T5ForConditionalGeneration
from transformers.models.bert import BertModel, BertTokenizer
from src.AttentionMech import AttentionMech
from src.LayerSetup2 import EventLayers, T5Layers
from src.TimesDates import TimesDates

class DecodeOutput(AttentionMech):
    def __init__(self, input_text, input_size=768, hidden_size=512, output_size=100, t5_vocab_size=32128):
        super().__init__(data=None, model_name='bert-base-uncased')
        self.input_text = input_text
        self.event_layers = EventLayers(input_size, hidden_size, output_size)
        self.t5_layers = T5Layers(output_size, input_size, t5_vocab_size)


    def process_text(self):
        bert_ids, bert_attention, bert_input_embeddings = self.text_embeddings(self.input_text, is_t5=False)
        t5_ids, t5_attention, t5_input_embeddings = self.text_embeddings(self.input_text, is_t5=True)
        tag_ids, tag_attention, input_tag_embeddings = self.bert_tokenizer(self.input_text, return_tensors='pt', padding='max_length', truncation=True, max_length=100)

        return bert_input_embeddings, t5_input_embeddings, input_tag_embeddings

    def forward(self):
        bert_input_embeddings, t5_input_embeddings, input_tag_embeddings = self.process_text()
        
        combined_features = torch.cat((input_tag_embeddings, bert_input_embeddings), dim=1)
        event_output = self.event_layers.sequential(combined_features)

        

    
from src.LayerSetup2 import EventLayers, T5Layers
import torch
import torch.nn as nn
from transformers.models.t5 import T5Tokenizer

class ForwardSetup(nn.Module):
    def __init__(self, input_size=768, hidden_size=512, output_size=100, t5_vocab_size=32128):
        super().__init__()
        self.event_layers = EventLayers(input_size, hidden_size, output_size)
        self.t5_layers = T5Layers(output_size, input_size, t5_vocab_size)
        self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')

    def get_token_ids(self, embeddings):
        logits = self.t5_layers.embeddings_to_logits(embeddings)
        token_ids = torch.argmax(logits, dim=-1)
        token_ids = token_ids.long()
        return logits, token_ids
    
    def concatenate_tokens_ids(self, token_ids, sep_value=102):
        batch_size = token_ids[0].shape[0]
        seperator = torch.full((batch_size, 1), sep_value, dtype=torch.long)
        
        merged_token_ids = torch.cat([
            token_ids[0], seperator, token_ids[1], seperator, token_ids[2], seperator, token_ids[3]
        ], dim=1)
        return merged_token_ids

    def forward(self, 
                tag_embeds, bert_input_embeds, t5_input_embeds, 
                time_embeddings, date_embeddings, response_embedding, 
                t5_event_embeddings):
        """
        Performs a forward pass through the model, combining tag and input embeddings, processing them through event and T5 layers, and returning the outputs.
        Args:
            tag_embeds (torch.Tensor): Embeddings representing tags, shape (batch_size, tag_dim).
            input_embeds (torch.Tensor): Embeddings representing input features, shape (batch_size, input_dim).
            response_embedding (torch.Tensor): Embeddings representing the target response, used for decoder input and labels.
        Returns:
            dict: A dictionary containing:
                - 'event_output' (torch.Tensor): Output from the event processing layers.
                - 't5_output' (transformers.modeling_outputs.Seq2SeqLMOutput): Output from the T5 model, including loss and logits.
        Raises:
            Exception: If any error occurs during the forward pass, it is printed and re-raised.
        """

        try:
            combined_features = torch.cat((tag_embeds, bert_input_embeds), dim=1)    
            event_output = self.event_layers.event_sequential(combined_features)
            intent_output = self.event_layers.intent_sequential(combined_features)
            
            t5_projection = self.t5_layers.t5_projection(event_output)
            event_logits, event_ids = self.get_token_ids(t5_event_embeddings)
            text_logits, text_ids = self.get_token_ids(t5_input_embeds)
            time_logits, time_ids = self.get_token_ids(time_embeddings)
            date_logits, date_ids = self.get_token_ids(date_embeddings)
            t5_input_ids = self.concatenate_tokens_ids([event_ids, text_ids, time_ids, date_ids])
            t5_attention_mask = torch.ones_like(t5_input_ids, dtype=torch.long)

            response_logits, label_ids = self.get_token_ids(response_embedding)
            t5_output = self.t5_layers.t5_model(
                input_ids=t5_input_ids,
                attention_mask=t5_attention_mask,
                labels=label_ids,
            )      

            return {
                'event_output': event_output,
                'intent_output': intent_output,
                't5_output': t5_output,
            }        
        except Exception as e:
            print(f"Error during forward pass: {e}")
            raise
from src.LayerSetup2 import EventLayers, T5Layers
import torch
import torch.nn as nn
from transformers.models.t5 import T5Tokenizer

class ForwardSetup(nn.Module):
    def __init__(self, input_embedding_size=768, input_pos_size=1, hidden_size=512, output_size=1, t5_vocab_size=32128):
        super().__init__()
        self.event_layers = EventLayers(input_embedding_size, input_pos_size, hidden_size, output_size)
        self.t5_layers = T5Layers(output_size, input_embedding_size, t5_vocab_size)
        self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')

    def concatenate_tokens_ids(self, token_ids, sep_value=102):
        batch_size = token_ids[0].shape[0]
        seperator = torch.full((batch_size, 1), sep_value, dtype=torch.long)
        
        merged_token_ids = torch.cat([
            token_ids[0], seperator, token_ids[1], seperator, token_ids[2], seperator, token_ids[3]
        ], dim=1)
        return merged_token_ids

    def forward(self, 
                tag_embeds, bert_input_embeds, t5_input_ids,
                response_ids, t5_event_ids, t5_time_ids, t5_date_ids):
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
            input_embeds_event = self.event_layers.get_embedding_sequential(bert_input_embeds)
            input_embeds_intent = self.event_layers.get_embedding_sequential(bert_input_embeds)
            input_pos_event = self.event_layers.get_pos_sequential(tag_embeds)
            input_pos_intent = self.event_layers.get_pos_sequential(tag_embeds)
            event_output = self.event_layers.event_combined(torch.cat([input_embeds_event, input_pos_event], dim=-1))
            intent_output = self.event_layers.intent_sequential(torch.cat([input_embeds_intent, input_pos_intent], dim=-1))

            t5_input_ids = self.concatenate_tokens_ids([t5_input_ids, t5_event_ids, t5_time_ids, t5_date_ids])
            t5_attention_mask = torch.ones_like(t5_input_ids, dtype=torch.long)

            t5_output = self.t5_layers.t5_model(
                input_ids=t5_input_ids,
                attention_mask=t5_attention_mask,
                labels=response_ids,
            )

            return {
                'event_logits': event_output,  # shape [batch, seq_len, 100]
                'intent_output': intent_output,
                't5_output': t5_output,
            }        
        except Exception as e:
            print(f"Error during forward pass: {e}")
            raise
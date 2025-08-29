from src.LayerSetup2 import EventLayers, T5Layers
import torch
import torch.nn as nn
from transformers.models.t5 import T5Tokenizer

class ForwardSetup(nn.Module):
    def __init__(self, input_embedding_size=768, hidden_size=512, output_size=768):
        super().__init__()
        self.event_layers = EventLayers(input_embedding_size, hidden_size, output_size)
        self.intent_layers = EventLayers(input_embedding_size, hidden_size, output_size)
        self.t5_layers = T5Layers()

    def forward(self, bert_input_embeds, t5_input_ids, t5_attention_mask, response_ids):
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
            event_output = self.event_layers.event_sequential(bert_input_embeds)
            intent_output = self.intent_layers.intent_sequential(bert_input_embeds)

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
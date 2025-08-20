import pandas as pd
from transformers.models.bert import BertModel, BertTokenizer
from transformers.models.t5 import T5Tokenizer, T5Model
from sklearn.preprocessing import MinMaxScaler
import spacy 
import os
import torch
import numpy as np
import math

files = [
    "data/procesed_tensors(1).pt",
    "data/procesed_tensors(2).pt",
    "data/procesed_tensors(3).pt",
    "data/procesed_tensors(4).pt",
    "data/procesed_tensors(5).pt",
    "data/procesed_tensors(6).pt",
    "data/procesed_tensors(7).pt",
    "data/procesed_tensors(8).pt",
    "data/procesed_tensors(9).pt",
    "data/procesed_tensors(10).pt",
]

pos_tags = {
    "ADJ": 0.5, "ADP": 1, "ADV": 0.2, "AUX": 0.2, "CCONJ": 0, "DET": 1,
    "INTJ": 0, "NOUN": 2, "NUM": 1, "PART": 0.3, "PRON": 1, "PROPN": 2,
    "PUNCT": 0, "SCONJ": 0, "SYM": 0, "VERB": 2, "X": 0
}

class AttentionMech:
    def __init__(self, data, model_name='bert-base-uncased', filename="data/event_scheduling.csv"):
        self.data = pd.DataFrame(data, columns=["input_text", "response_text", "type", "event_name", "event_time", "event_date"])
        self.model_name = model_name
        self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.t5_model = T5Model.from_pretrained('t5-base')
        self.t5_model.eval()
        self.bert_tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert_model = BertModel.from_pretrained(model_name)
        self.bert_model.eval()
        self.nlp = spacy.load("en_core_web_sm")
        self.filename = filename

        self.files = files
        self.pos_tags = pos_tags
        for file in self.files:
            if not os.path.exists(file): #TODO: Add more for intention, time, and date
                dataset = {
                    "bert_input_embeddings": torch.empty((0, 100, 768), dtype=torch.float32),
                    "t5_input_ids": torch.empty((0, 100), dtype=torch.long),
                    "t5_attention_mask": torch.empty((0, 100), dtype=torch.long),
                    "tag_embeddings": torch.empty((0, 100, 1), dtype=torch.float32),
                    "response_ids": torch.empty((0, 100), dtype=torch.long),
                    "priority_scores": torch.empty((0, 100, 768), dtype=torch.float32),
                    "intention_scores": torch.empty((0, 768), dtype=torch.float32)
                }
                torch.save(dataset, file)
        self.broken_text = []

#MARK: Embeddings
    def text_embeddings(self, data, is_t5=False): 
        input_doc = self.nlp(data)
        input_tokens = [token.lemma_ for token in input_doc if not token.is_stop and not token.is_punct]
        input_text = " ".join(input_tokens)
        if is_t5:
            inputs = self.t5_tokenizer(input_text, return_tensors='pt', padding='max_length', truncation=True, max_length=100)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            outputs = self.t5_model.encoder(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                return_dict=True
            )
        else:
            inputs = self.bert_tokenizer(input_text, return_tensors='pt', padding='max_length', truncation=True, max_length=100)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            outputs = self.bert_model(**inputs)

        return input_ids, attention_mask, outputs.last_hidden_state
    
    def tag_embeddings(self, input_text):
        text_doc = self.nlp(input_text.lower())
        input_text = " ".join([token.lemma_ for token in text_doc if not token.is_stop and not token.is_punct])
        processed_doc = self.nlp(input_text)
        tags = " ".join([token.pos_ for token in processed_doc])
        padded_tags = "[CLS] " + tags + " [SEP]" + " [PAD]" * (100 - (len(tags.split(' ')) + 2))
        label_encodings = self.label_encoding(padded_tags)
        return torch.tensor([label_encodings], dtype=torch.float32)
    
    def label_encoding(self, padded_tags):
        tokens = padded_tags.split(' ')
        labels = []
        for token in tokens:
            if token in self.pos_tags:
                labels.append([self.pos_tags[token]])
            else:
                labels.append([0])
        return labels

#MARK: Attention Scores
    def generate_scores(self, input_tokens, input_embedding, event):
        try:
            event_doc = self.nlp(event.lower())
            raw_event_tokens = [token.lemma_ for token in event_doc if not token.is_stop and not token.is_punct]
            event_tokens = []
            event_index = 0
            for index in range(len(input_tokens)):
                if event_index < len(raw_event_tokens):
                    if input_tokens[index] == raw_event_tokens[event_index]:
                        event_tokens.append(raw_event_tokens[event_index])
                        event_index += 1
                    else:
                        event_tokens.append("[PAD]")
                else:
                    event_tokens.append("[PAD]")

            bert_event_tokens = self.bert_tokenizer(" ".join(event_tokens), return_tensors='pt', padding='max_length', truncation=True, max_length=100)
            event_embeddings = self.bert_model(input_ids=bert_event_tokens['input_ids'], attention_mask=bert_event_tokens['attention_mask']).last_hidden_state

            priority_scores = self.apply_dot_product(event_embeddings, input_embedding, input_embedding)
            return priority_scores
        except Exception as e:
            print(f"Error generating scores for input text: {" ".join(input_tokens)} and event {event}. Error: {e}")
            self.broken_text.append(" ".join(input_tokens))
            return None
        
    def get_intention_scores(self, intention, input_embeddings):
        intention_doc = self.nlp(intention.lower())
        intention_string = " ".join([token.lemma_ for token in intention_doc if not token.is_stop and not token.is_punct])
        inputs = self.bert_tokenizer(intention_string, return_tensors='pt', padding='max_length', truncation=True, max_length=100)
        intention_embeddings = self.bert_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).last_hidden_state
        try:
            intention_scores = self.apply_dot_product(intention_embeddings, input_embeddings, input_embeddings)
            return intention_scores.mean(dim=1) if intention_scores is not None else None
        except Exception as e:      
            print(f"Error generating intention scores for {intention}. Error: {e}")
            return None
        
    def apply_dot_product(self, key, value, query):
            try: 
                scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(768)
                scaled_scores = torch.softmax(scores, dim=-1)
                return torch.matmul(scaled_scores, value)
            except Exception as e:
                print(f"Error applying dot product: {e}")
                return None
#MARK: Main Processing
    def generate_data(self, batch_size=100):
        current_file_index = 0
        total_samples = len(self.data)

        for i in range(0, total_samples, batch_size):
            batch_end = min(i + batch_size, total_samples)
            batch_data = []
            for j in range(i, batch_end):
                row = self.data.iloc[j]
                input_text = row['input_text']
                response_text = row['response_text']
                event_text = row["event_name"]
                t5_training_text = "RESPOND TO THIS REQUEST: " + input_text + ". EVENT: " + event_text + ". INTENTION: " + row['type'] + ". TIME: " + row['event_time'] + ". DATE: " + row['event_date']

                input_doc = self.nlp(input_text.lower())
                input_tokens = [token.lemma_ for token in input_doc if not token.is_stop and not token.is_punct]

                bert_input_ids, bert_attention_mask, bert_embeddings = self.text_embeddings(" ".join(input_tokens), is_t5=False)
                t5_input_ids, t5_attention_mask, t5_embeddings = self.text_embeddings(t5_training_text, is_t5=True)             
                tag_embeddings = self.tag_embeddings(" ".join(input_tokens))
                priority_scores = self.generate_scores(input_tokens, bert_embeddings, event_text)
                intention_scores = self.get_intention_scores(row['type'], bert_embeddings)

                if priority_scores is None or intention_scores is None:
                    print(f"Skipping row {j} due to broken text: {input_text}")
                    continue
                response_ids, response_attention_mask, response_embeddings = self.text_embeddings("TRUE RESPONSE: " + response_text, is_t5=True)
                row = {
                    "bert_input_embeddings": bert_embeddings.squeeze(0),
                    "t5_input_ids": t5_input_ids.squeeze(0),
                    "t5_attention_mask": t5_attention_mask.squeeze(0),
                    "tag_embeddings": tag_embeddings.squeeze(0),
                    "response_ids": response_ids.squeeze(0),
                    "priority_scores": priority_scores.squeeze(0),
                    "intention_scores": intention_scores.squeeze(0)
                }
                batch_data.append(row)
                print(f"\rRow {j + 1}/{total_samples} processed", end='', flush=True)

            batch_tensors = {}
            for key in batch_data[0]:
                batch_tensors[key] = torch.stack([item[key] for item in batch_data])

            dataset = torch.load(self.files[current_file_index])
            for key in batch_tensors.keys():
                dataset[key] = torch.cat([dataset[key], batch_tensors[key]], dim=0)

            if dataset['bert_input_embeddings'].shape[0] >= 1500:
                torch.save(dataset, self.files[current_file_index])
                current_file_index += 1
                print(f"Saved {self.files[current_file_index]} with shape {dataset['bert_input_embeddings'].shape}")
                if current_file_index >= len(self.files):
                    print("All files processed.")
                    break
            else:
                torch.save(dataset, self.files[current_file_index])
                print(f"Saved {self.files[current_file_index]}")

            del batch_data, batch_tensors
            torch.cuda.empty_cache() if torch.cuda.is_available() else None 
        print("Data generation completed.")
    
    def save_broken_text(self):
        if self.broken_text:
            with open("data/broken_text.txt", "w") as f:
                for text in self.broken_text:
                    f.write(text + "\n")
            print(f"Saved {len(self.broken_text)} broken texts to data/broken_text.txt")
        else:
            print("No broken texts found.")

if __name__ == "__main__":
    data = pd.read_csv("data/event_scheduling.csv")
    attention_mech = AttentionMech(data)
    attention_mech.generate_data()
    attention_mech.save_broken_text()

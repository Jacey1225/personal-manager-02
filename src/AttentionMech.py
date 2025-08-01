import pandas as pd
from transformers.models.bert import BertModel, BertTokenizer
from transformers.models.t5 import T5Tokenizer, T5Model
import spacy 
import os
import torch

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
        self.embedding_layer = torch.nn.Embedding(100, 768)
        self.filename = filename

        self.files = files
        self.pos_tags = pos_tags
        for file in self.files:
            if not os.path.exists(file): #TODO: Add more for intention, time, and date
                dataset = {
                    "bert_input_embeddings": torch.empty((0, 100, 768), dtype=torch.float32),
                    "t5_input_embeddings": torch.empty((0, 100, 768), dtype=torch.float32),
                    "tag_embeddings": torch.empty((0, 100, 1), dtype=torch.float32),
                    "response_embeddings": torch.empty((0, 100, 768), dtype=torch.float32),
                    "priority_scores": torch.empty((0, 100), dtype=torch.float32),
                    "t5_event_embeddings": torch.empty((0, 100, 768), dtype=torch.float32),
                    "intention_scores": torch.empty((0, 1), dtype=torch.float32),
                    "intent_embeddings": torch.empty((0, 100, 768), dtype=torch.float32),
                    "time_embeddings": torch.empty((0, 100, 768), dtype=torch.float32),
                    "date_embeddings": torch.empty((0, 100, 768), dtype=torch.float32)
                }
            torch.save(dataset, file)
        self.broken_text = []

    def text_embeddings(self, data, is_t5=False): 
        if is_t5:
            inputs = self.t5_tokenizer(data, return_tensors='pt', padding='max_length', truncation=True, max_length=100)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            outputs = self.t5_model.encoder(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                return_dict=True
            )
        else:
            inputs = self.bert_tokenizer(data, return_tensors='pt', padding='max_length', truncation=True, max_length=100)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            outputs = self.bert_model(**inputs)

        return input_ids, attention_mask, outputs.last_hidden_state
    
    def label_encoding(self, padded_tags):
        tokens = padded_tags.split(' ')
        labels = []
        for token in tokens:
            if token in self.pos_tags:
                labels.append([self.pos_tags[token]])
            else:
                labels.append([0])
        return labels
    
    def tag_embeddings(self, input_text):
        text_doc = self.nlp(input_text.lower())
        input_text = " ".join([token.lemma_ for token in text_doc if not token.is_stop and not token.is_punct])
        processed_doc = self.nlp(input_text)
        tags = " ".join([token.pos_ for token in processed_doc])
        padded_tags = "[CLS] " + tags + " [SEP]" + " [PAD]" * (100 - (len(tags.split(' ')) + 2))
        label_encodings = self.label_encoding(padded_tags)
        return torch.tensor([label_encodings], dtype=torch.float32)

    def generate_scores(self, input_text, event_text): 
        tokens = self.bert_tokenizer.tokenize(input_text)
        tokens = ['[CLS]'] + tokens
        if len(tokens) < 99:
            tokens = tokens + ['[SEP]']
        else:
            tokens = tokens[:99] + ['[SEP]']
        if len(tokens) < 100:
            tokens += ['[PAD]'] * (100 - len(tokens))
        else:
            tokens = tokens[:100]

        event_start = input_text.lower().find(event_text.lower())
        event_end = event_start + len(event_text)
        if event_start == -1:
            self.broken_text.append(input_text)
            return None

        char_idx = 0
        scores = []
        for tok in tokens:
            if tok in ['[CLS]', '[SEP]', '[PAD]']:
                scores.append(0.0)
            else:
                clean_tok = tok.replace('##', '')
                idx = input_text.lower().find(clean_tok, char_idx)
                if idx == -1:
                    scores.append(0.0)
                else:
                    if idx >= event_start and idx < event_end:
                        scores.append(1.0)
                    else:
                        scores.append(0.0)
                    char_idx = idx + len(clean_tok)
        if len(scores) < 100:
            scores += [0.0] * (100 - len(scores))
        else:
            scores = scores[:100]
        return torch.tensor([scores], dtype=torch.float32)

    def get_intention_scores(self, intention):
        intention_score = 0.0
        if intention.lower() == "delete":
            intention_score = -1.0
        elif intention.lower() == "add":
            intention_score = 1.0
        elif intention.lower() == "modify":
            intention_score = 0.0
        return torch.tensor([intention_score], dtype=torch.float32)

    def generate_data(self, batch_size=25):
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
    
                tag_embeddings = self.tag_embeddings(input_text)
                bert_input_ids, bert_attention_mask, bert_embeddings = self.text_embeddings(input_text, is_t5=False)
                t5_input_ids, t5_attention_mask, t5_embeddings = self.text_embeddings("RESPOND TO THIS REQUEST: " + input_text, is_t5=True)
                t5_event_ids, t5_event_attention_mask, t5_event_embeddings = self.text_embeddings(("EVENT: " + event_text), is_t5=True)
                intent_input_ids, intent_attention_mask, intent_embeddings = self.text_embeddings(("INTENTION: " + row['type']), is_t5=True)
                time_input_ids, time_attention_mask, time_embeddings = self.text_embeddings(("TIME: " + row['event_time']), is_t5=True)
                date_input_ids, date_attention_mask, date_embeddings = self.text_embeddings(("DATE: " + row['event_date']), is_t5=True)
                priority_scores = self.generate_scores(input_text, event_text)
                intention_scores = self.get_intention_scores(row['type'])
                if priority_scores is None:
                    print(f"Skipping row {j} due to broken text: {input_text}")
                    continue
                response_ids, response_attention_mask, response_embeddings = self.text_embeddings("TRUE RESPONSE: " + response_text, is_t5=True)
                row = {
                    "bert_input_embeddings": bert_embeddings.squeeze(0),
                    "t5_input_embeddings": t5_embeddings.squeeze(0),
                    "tag_embeddings": tag_embeddings.squeeze(0),
                    "response_embeddings": response_embeddings.squeeze(0),
                    "priority_scores": priority_scores.squeeze(0),
                    "t5_event_embeddings": t5_event_embeddings.squeeze(0), 
                    "intention_scores": intention_scores,
                    "intent_embeddings": intent_embeddings.squeeze(0),
                    "time_embeddings": time_embeddings.squeeze(0),
                    "date_embeddings": date_embeddings.squeeze(0)
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

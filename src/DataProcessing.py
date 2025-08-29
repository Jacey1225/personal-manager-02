import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from transformers.models.t5 import T5Tokenizer
import spacy
import os

class DataProcessor:
    def __init__(self, filename='data/event_scheduling.csv', tokenizer_name='t5-small', max_length=250, batch_size=75):
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.data = pd.read_csv(filename)
        self.data_size = len(self.data)
        self.batch_size = batch_size
        self.nlp = spacy.load("en_core_web_sm")
        self.processed_data_filename = 'data/processed_event_data'
        self.num_files = 0

    def lemmatize_row(self, row):
        lemmatized_row = []
        for item in row:
            doc = self.nlp(item)
            lemmatized_as_list = [token.lemma_ for token in doc if not token.is_stop or token.is_punct]
            lemmatized_text = " ".join(lemmatized_as_list)
            lemmatized_row.append(lemmatized_text)
        
        return lemmatized_row

    def row_processed(self, row):
        row_input_ids = []
        row_attention_masks = []
        for item in row:
            encoded_dict = self.tokenizer(
                item,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            row_input_ids.append(encoded_dict['input_ids'])
            row_attention_masks.append(encoded_dict['attention_mask'])

        return row_input_ids, row_attention_masks
    
    def save_batch(self, input_ids, attention_masks):
        dataset = torch.load(self.processed_data_filename + '.pt') if os.path.exists(self.processed_data_filename + '.pt') else []
        dataset.append((input_ids, attention_masks))
        torch.save(dataset, self.processed_data_filename + '.pt')
    
    def process_data(self):
        for i in range(0, self.data_size, self.batch_size):
            batch_data = self.data.iloc[i:i+self.batch_size]
            lemmatized_batch = batch_data.apply(self.lemmatize_row, axis=1)
            input_ids_list = []
            attention_masks_list = []
            for _, row in lemmatized_batch.iterrows(): #type: ignore
                row_input_ids, row_attention_masks = self.row_processed(row)
                input_ids_list.append(torch.cat(row_input_ids, dim=0).unsqueeze(0))
                attention_masks_list.append(torch.cat(row_attention_masks, dim=0).unsqueeze(0))

            batch_input_ids = torch.cat(input_ids_list, dim=0)
            batch_attention_masks = torch.cat(attention_masks_list, dim=0)
            if os.path.getsize(self.processed_data_filename + '.pt') > 1e+9:
                self.num_files += 1
                self.processed_data_filename = self.processed_data_filename + "_" + str(self.num_files)

            self.save_batch(batch_input_ids, batch_attention_masks)
            print(f"\rProcessed batch {i//self.batch_size + 1}/{(self.data_size + self.batch_size - 1)//self.batch_size}", end='', flush=True)

if __name__ == "__main__":
    processor = DataProcessor()
    processor.process_data()
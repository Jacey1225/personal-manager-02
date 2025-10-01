from transformers import T5ForConditionalGeneration
import torch
import pandas as pd
import os
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.trainer_seq2seq import Seq2SeqTrainer
from pydantic import BaseModel
from torch.utils.data import random_split, Dataset
import proxy_bypass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingConfig(BaseModel):
    model_name: str = 't5-small'
    output_dir: str = './model'
    num_train_epochs: int = 20
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 5e-04
    weight_decay: float = 0.01
    logging_dir: str = './logs'
    logging_steps: int = 10
    evaluation_strategy: str = 'epoch'
    save_strategy: str = 'epoch'
    save_total_limit: int = 2
    predict_with_generate: bool = True
    fp16: bool = False

class TrainingData(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

class ModelTrainer:
    def __init__(self, config: TrainingConfig):
        proxy_bypass._configure_proxy_bypass()  # Bypass proxy for Huggingface
        self.config = config
        self.model = T5ForConditionalGeneration.from_pretrained(self.config.model_name)
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)
        if not os.path.exists(self.config.logging_dir):
            os.makedirs(self.config.logging_dir)

    def load_data(self, data_path: str='data/processed_event_data.pt'):
        """Loads the training data from a specified file.

        Args:
            data_path (str, optional): The path to the data file. Defaults to 'data/processed_event_data.pt'.

        Raises:
            FileNotFoundError: If the data file does not exist.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The file '{data_path}' does not exist.")
        data = torch.load(data_path)
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        labels = data['labels']

        dataset = TrainingData(input_ids, attention_mask, labels)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])
        logger.info(f"Loaded dataset with {len(self.train_dataset)} training samples and {len(self.val_dataset)} validation samples.")

    def train(self):
        """Trains the model on the loaded dataset.
        """
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            logging_dir=self.config.logging_dir,
            logging_steps=self.config.logging_steps,
            eval_strategy=self.config.evaluation_strategy,
            save_strategy=self.config.save_strategy,
            save_total_limit=self.config.save_total_limit,
            predict_with_generate=self.config.predict_with_generate,
            fp16=self.config.fp16,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
        )

        trainer.train()
        logger.info("Training complete.")
        self.model.save_pretrained(self.config.output_dir)
        logger.info(f"Model saved to {self.config.output_dir}")

if __name__ == "__main__":
    config = TrainingConfig()
    trainer = ModelTrainer(config)
    trainer.load_data()
    trainer.train()
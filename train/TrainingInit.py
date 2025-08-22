import torch
import torch.nn as nn
from src.ModelSetup2 import ForwardSetup
from transformers.optimization import get_scheduler

class TrainingInit(nn.Module):
    def __init__(self, 
                 batch_size, epochs, event_lr, intent_lr, t5_lr, current_file, full_data_size, 
                 training_size=0.8, validation_size=0.1, testing_size=0.1, 
                 current_file_index=0, last_file_index=10):
        super().__init__()
        self.batch_size = batch_size
        self.epochs = epochs
        self.current_file = current_file
        self.current_file_index = current_file_index
        self.last_file_index = last_file_index
        self.file_data = torch.load(self.current_file, map_location=torch.device('cpu'))
        self.forward_setup = ForwardSetup()
        self.event_layers = self.forward_setup.event_layers
        self.intent_layers = self.forward_setup.intent_layers
        self.t5_layers = self.forward_setup.t5_layers

        self.event_optimizer = torch.optim.AdamW(self.event_layers.parameters(), lr=event_lr)
        self.intent_optimizer = torch.optim.AdamW(self.intent_layers.parameters(), lr=intent_lr)
        self.t5_optimizer = torch.optim.AdamW(self.t5_layers.parameters(), lr=t5_lr)
        self.t5_scheduler = get_scheduler(
            "linear",
            optimizer=self.t5_optimizer,
            num_warmup_steps=150,
            num_training_steps=self.epochs * (full_data_size // self.batch_size)
        )

        self.full_data_size = full_data_size
        self.file_size = self.file_data['bert_input_embeddings'].shape[0]

        self.training_size = training_size
        self.validation_size = validation_size
        self.testing_size = testing_size
        self.num_training_files = int(self.last_file_index * self.training_size)
        self.num_validation_files = (self.last_file_index - self.num_training_files) // 2
        self.num_testing_files = (self.last_file_index - self.num_training_files) // 2

    def load_new_file(self, file_index):
        self.current_file = f"data/procesed_tensors({file_index + 1}).pt"
        self.file_data = torch.load(self.current_file, map_location=torch.device('cpu'))
        self.file_size = self.file_data['priority_scores'].shape[0]
        print(f"Loaded file {self.current_file} with size {self.file_size}")

    def get_data_splits(self):
        """
        Splits the loaded file data into training, validation, and testing sets based on the current counts of training, validation, and testing files.
        Returns:
            tuple: A tuple containing three dictionaries (training, validation, testing), each mapping file keys to their respective data splits.
        Notes:
            - If there are training files, all file data is assigned to the training set and the training file count is decremented.
            - If the number of validation files equals the number of testing files, the file data is split in half: the first half goes to validation, the second half to testing.
            - If there are remaining validation files, all file data is assigned to the validation set and the validation file count is decremented.
            - Otherwise, the remaining data (after training files) is assigned to the testing set and the testing file count is decremented.
        """

        training = {}
        validation = {}
        testing = {}
        if self.num_training_files > 0:
            for key, value in self.file_data.items():
                training[key] = value
            self.num_training_files -= 1
        if self.num_validation_files == self.num_testing_files and self.num_training_files == 0:
            split_index = self.file_size // 2
            for key, value in self.file_data.items():
                validation[key] = value[:split_index]
                testing[key] = value[split_index:]
            self.num_validation_files -= 1
            self.num_testing_files -= 1
        else:
            if self.num_validation_files > 0 and self.num_training_files == 0:
                for key, value in self.file_data.items():
                    validation[key] = value
                self.num_validation_files -= 1
            else:
                if self.num_training_files == 0:
                    for key, value in self.file_data.items():
                        testing[key] = value[self.num_training_files:self.file_size]
                    self.num_testing_files -= 1
        
        print(f"Last File Index: {self.last_file_index}, Training Size: {self.training_size}")
        print(f"Remaining Files: Training: {self.num_training_files}, Validation: {self.num_validation_files}, Testing: {self.num_testing_files}")
        return training, validation, testing


import torch
from src.ModelSetup2 import ForwardSetup
from train.TrainingInit import TrainingInit
import matplotlib.pyplot as plt
import numpy as np

class Training(TrainingInit):
    def __init__(self, batch_size, epochs, event_lr, t5_lr):
        super().__init__(batch_size, epochs, event_lr, t5_lr, current_file="data/procesed_tensors(1).pt", full_data_size=15000, last_file_index=10)
        self.forward_setup = ForwardSetup()
        self.current_file_index = 0
        self.epochs = epochs
        self.batch_size = batch_size
        self.training_data = None
        self.validation_data = None
        self.testing_data = None
        self.full_training_length = int(self.full_data_size * self.training_size)
        self.event_losses = []
        self.intent_losses = []
        self.t5_losses = []
        plt.figure(figsize=(20, 8)) 
        figmanager = plt.get_current_fig_manager()
        figmanager.full_screen_toggle()  # type: ignore

    def update_graph(self):
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.plot(range(len(self.event_losses)), self.event_losses, label='Event Loss', color='red', linewidth=2)
        plt.plot(range(len(self.intent_losses)), self.intent_losses, label='Intent Loss', color='green', linewidth=2)
        plt.title('Event/Intent Loss Over Data Size')
        plt.xlabel('Data Size')
        plt.ylabel('Event/Intent Loss')
        plt.legend(fontsize=10)
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(range(len(self.t5_losses)), self.t5_losses, label='T5 Loss', color='orange', linewidth=2)
        plt.title('T5 Loss Over Data Size')
        plt.xlabel('Data Size')
        plt.ylabel('T5 Loss')
        plt.legend(fontsize=10)
        plt.grid(True)

        plt.tight_layout()
        plt.pause(0.1)
        plt.show(block=False)
        
    def fetch_batch_data(self, i):
        if self.training_data is not None:
            tag_embeddings_set = self.training_data['tag_embeddings'][i:i + self.batch_size]
            bert_input_embeddings_set = self.training_data['bert_input_embeddings'][i:i + self.batch_size]
            t5_input_embeddings_set = self.training_data['t5_input_embeddings'][i:i + self.batch_size]
            response_embeddings_set = self.training_data['response_embeddings'][i:i + self.batch_size]
            t5_event_embeddings_set = self.training_data['t5_event_embeddings'][i:i + self.batch_size]
            time_embeddings_set = self.training_data['time_embeddings'][i:i + self.batch_size]
            date_embeddings_set = self.training_data['date_embeddings'][i:i + self.batch_size]
            priority_scores_set = self.training_data['priority_scores'][i:i + self.batch_size]
            intention_scores_set = self.training_data['intention_scores'][i:i + self.batch_size]

        return (tag_embeddings_set, bert_input_embeddings_set, t5_input_embeddings_set,
                response_embeddings_set, t5_event_embeddings_set, 
                time_embeddings_set, date_embeddings_set, priority_scores_set, intention_scores_set)
    def train(self):
        """
        Trains the model for a specified number of epochs using the provided training data.
        The method performs the following steps:
        - Sets the model layers to training mode.
        - Iterates through the specified number of epochs.
        - Splits the data into training, validation, and testing sets.
        - Processes the training data in batches:
            - Zeroes gradients for both event and T5 layers.
            - Loads new data files as needed when the current batch exceeds available data.
            - Computes outputs using the forward pass.
            - Calculates and accumulates event and T5 losses.
            - Performs backpropagation and optimizer steps for both event and T5 layers.
            - Prints batch progress and loss values.
        - Computes and prints average losses per epoch.
        - Validates the model after each epoch if validation data is available.
        - Stops training when all files are processed or no more training data is available.
        Prints progress and loss information throughout training.
        """

        self.forward_setup.event_layers.train()
        self.forward_setup.t5_layers.train()

        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs} for {self.full_training_length} training samples")
            average_loss = {
                'event_loss': 0.0,
                'intent_loss': 0.0,
                't5_loss': 0.0
            }

            self.num_training_files = int(self.last_file_index * self.training_size)
            self.num_validation_files = (self.last_file_index - self.num_training_files) // 2
            self.num_testing_files = (self.last_file_index - self.num_training_files) // 2
            self.training_data, self.validation_data, self.testing_data = self.get_data_splits()
            index = 0
            self.current_file_index = 0
            for i in range(0, self.full_training_length, self.batch_size):
                if index + self.batch_size >= len(self.training_data['bert_input_embeddings']):
                    self.current_file_index += 1
                    print(f"\n{index + self.batch_size} > {len(self.training_data['bert_input_embeddings'])} at index = {index}. Loading next file. Next file index: {self.current_file_index}")
                    self.load_new_file(self.current_file_index)
                    self.training_data, self.validation_data, self.testing_data = self.get_data_splits()
                    index = 0
                    if self.training_data is None:
                        print('No more training data available.')
                        break
                    continue
                self.event_layers.zero_grad()
                self.t5_layers.zero_grad()
                tag_embeddings_set, bert_input_embeddings_set, t5_input_embeddings_set, response_embeddings_set, t5_event_embeddings_set, time_embeddings_set, date_embeddings_set, priority_scores_set, intention_scores_set = self.fetch_batch_data(index)

                outputs = self.forward_setup.forward(tag_embeddings_set, bert_input_embeddings_set, t5_input_embeddings_set, time_embeddings_set, date_embeddings_set, response_embeddings_set, t5_event_embeddings_set)
                event_output = outputs['event_output'][:, -1, :]
                event_loss = self.forward_setup.event_layers.event_loss(event_output, priority_scores_set)
                event_loss.backward()
                self.event_optimizer.step()
                average_loss['event_loss'] += event_loss.item()
                self.event_losses.append(event_loss.item())

                intent_output = outputs['intent_output'][:, -1, :]
                intent_loss = self.forward_setup.event_layers.intent_loss(intent_output, intention_scores_set)
                intent_loss.backward()
                self.event_optimizer.step()
                average_loss['intent_loss'] += intent_loss.item()
                self.intent_losses.append(intent_loss.item())

                t5_loss = outputs['t5_output'].loss
                t5_loss.backward()
                self.t5_optimizer.step()
                average_loss['t5_loss'] += t5_loss.item()
                self.t5_losses.append(t5_loss.item())

                print(f"\rBatch {index // self.batch_size + 1}/{len(self.training_data['bert_input_embeddings']) // self.batch_size} processed | Event Loss: {event_loss.item():.4f},  Intent Loss: {intent_loss.item():.4f}, T5 Loss: {t5_loss.item():.4f}", end='', flush=True)
                index += self.batch_size
                self.update_graph()
            average_loss = {k: v / (self.full_training_length / self.batch_size) for k, v in average_loss.items()}

            if self.validation_data is not None:
                print(f"\nValidating after epoch {epoch + 1}... Validation Size: {self.validation_data['bert_input_embeddings'].shape[0] if self.validation_data else 0}")
                self.validate()
            else:
                print("No validation data available.")
                break
            print(f"\nEpoch {epoch + 1} completed. Average Loss: {average_loss}")

    def validate(self):
        """
        Validates the model using the provided validation data.
        This method sets the model to evaluation mode and computes the average event loss and T5 loss
        over the validation dataset in batches. It prints the validation losses at the end of the process.
        Returns:
            None if validation data is available and validation is performed.
            False if no validation data is available.
        """

        if self.validation_data is not None:
            self.forward_setup.eval()
            with torch.no_grad():
                validation_loss = {
                    'event_loss': 0.0,
                    't5_loss': 0.0
                }
                for i in range(0, len(self.validation_data['bert_input_embeddings']), self.batch_size):
                    tag_embeddings_set = self.validation_data['tag_embeddings'][i:i + self.batch_size]
                    bert_input_embeddings_set = self.validation_data['bert_input_embeddings'][i:i + self.batch_size]
                    t5_input_embeddings_set = self.validation_data['t5_input_embeddings'][i:i + self.batch_size]
                    time_embeddings_set = self.validation_data['time_embeddings'][i:i + self.batch_size]
                    date_embeddings_set = self.validation_data['date_embeddings'][i:i + self.batch_size]
                    response_embeddings_set = self.validation_data['response_embeddings'][i:i + self.batch_size]
                    event_embeddings_set = self.validation_data['t5_event_embeddings'][i:i + self.batch_size]

                    outputs = self.forward_setup.forward(tag_embeddings_set, bert_input_embeddings_set, t5_input_embeddings_set, time_embeddings_set, date_embeddings_set, response_embeddings_set, event_embeddings_set)
                    event_output = outputs['event_output'][:, -1, :]
                    event_loss = self.forward_setup.event_layers.event_loss(event_output, self.validation_data['priority_scores'][i:i + self.batch_size])
                    validation_loss['event_loss'] += event_loss.item()
                    
                    t5_loss = outputs['t5_output'].loss
                    validation_loss['t5_loss'] += t5_loss.item()

                validation_loss = {k: v / (len(self.validation_data['bert_input_embeddings']) / self.batch_size) for k, v in validation_loss.items()}
                print(f"Validation Loss: {validation_loss}")
        else:
            print("No validation data available.")
            return False
if __name__ == "__main__":
    training = Training(batch_size=32, epochs=15, event_lr=0.01, t5_lr=0.001)
    training.train()
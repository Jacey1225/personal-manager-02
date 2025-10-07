import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from proxy_bypass import _configure_proxy_bypass

class TestLLM:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The model path '{model_path}' does not exist.")
        _configure_proxy_bypass()
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.model.eval()
        
        self.test_inputs = {
            1: {
                "feature": "Event Time: 9am, Event Date: August 15th, Event Input: Can you please schedule a hiking trip at 9am on August 15th?",
                "label": "Event: Hiking Trip, Action: Schedule, Desired Response: Sure, I have scheduled your hiking trip for 9am on August 15th."
            },
            2: {
                "feature": "Event Time: None, Event Date: tomorrow, Event Input: Can you actually cancel my plans to workout tomorrow",
                "label": "Event: Workout, Action: Cancel, Desired Response: Your workout plans for tomorrow have been cancelled."
            },
            3: {
                "feature": "Event Time: 3pm, Event Date: None, Event Input: Lets change my pickleball tournament to 3pm instead of 5pm",
                "label": "Event: Pickleball Tournament, Action: Update, Desired Response: Your pickleball tournament has been rescheduled to 3pm."
            }
        }

    def evaluate_model(self):
        correct_predictions = 0
        total_predictions = len(self.test_inputs)

        for idx, test_case in self.test_inputs.items():
            inputs = self.tokenizer(test_case["feature"], return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=250)
            predicted_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            if predicted_text.strip() == test_case["label"].strip():
                correct_predictions += 1
            else:
                print(f"Test Case {idx} Failed:")
                print(f"Expected: {test_case['label']}")
                print(f"Got: {predicted_text}\n")

        accuracy = (correct_predictions / total_predictions) * 100
        print(f"Model Accuracy: {accuracy:.2f}%")
        return accuracy
    
if __name__ == "__main__":
    model_path = './model'  # Path to the trained model
    tester = TestLLM(model_path)
    tester.evaluate_model()
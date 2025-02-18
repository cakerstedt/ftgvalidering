from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import math
from datasets import load_dataset

# Ange rätt modellväg
MODEL_PATH = "models/gpt2_model"
CHECKPOINT_PATH = "models/gpt2_model/checkpoint-4"

# Ladda tokenizer från huvudmodellen
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)

# Ladda modellen från checkpoint
model = GPT2LMHeadModel.from_pretrained(CHECKPOINT_PATH)

# Ladda testdataset (justera efter ditt dataset)
test_dataset = load_dataset('text', data_files={'test': 'test_data.txt'})['test']

# Funktion för att utvärdera modellen
def evaluate_model(test_dataset, model):
    model.eval()
    total_loss = 0
    total_steps = 0
    
    for example in test_dataset:
        inputs = tokenizer.encode(example['text'], return_tensors="pt", max_length=512, truncation=True)
        labels = inputs.clone()

        with torch.no_grad():
            outputs = model(input_ids=inputs, labels=labels)
            loss = outputs.loss.item()
        
        total_loss += loss
        total_steps += 1
    
    avg_loss = total_loss / total_steps
    perplexity = math.exp(avg_loss)
    
    return avg_loss, perplexity

# Beräkna loss och perplexity
loss, perplexity = evaluate_model(test_dataset, model)

# Skriv ut resultat
print(f"Eval Loss: {loss:.4f}")
print(f"Perplexity: {perplexity:.4f}")

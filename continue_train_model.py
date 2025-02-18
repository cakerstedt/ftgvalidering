from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch
from datasets import Dataset
import os

# Ange sökvägen till den senaste checkpointen
CHECKPOINT_PATH = "models/gpt2_model/checkpoint-4"

# Ladda och förbered data från den bearbetade filen
def load_and_process_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        texts = f.readlines()  # Läser in varje rad som en egen textsträng
    return texts

def tokenize_function(examples):
    tokens = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()  # Lägg till labels för loss-beräkning
    return tokens

# Filväg för bearbetad data
DATA_FILE = "more_training_data.txt"

# Ladda dataset
train_texts = load_and_process_data(DATA_FILE)

# Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 har ingen standard pad-token, så vi sätter eos som padding

# Skapa Hugging Face Dataset
train_dataset = Dataset.from_dict({"text": train_texts}).map(tokenize_function, batched=True)

# Initiera GPT-2 modell
model = GPT2LMHeadModel.from_pretrained(CHECKPOINT_PATH)

# Träningsargument
training_args = TrainingArguments(
    output_dir="models/gpt2_model",
    evaluation_strategy="no",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    logging_dir="../logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Träna modellen
trainer.train()

# Spara modellen i en separat checkpoint-mapp
checkpoint_dir = f"models/gpt2_model/checkpoint-{trainer.state.global_step // 1000 + 1}"  # Sätter checkpoint-nummer baserat på träningssteg
model.save_pretrained(checkpoint_dir)
tokenizer.save_pretrained(checkpoint_dir)

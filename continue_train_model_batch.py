from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch
from datasets import Dataset, DatasetDict
import os

# Ange sökvägen till den senaste checkpointen
CHECKPOINT_PATH = "models/gpt2_model/checkpoint-1"

# Ladda och förbered data från den bearbetade filen (läs i batcher)
def load_and_process_data_in_batches(file_path, batch_size=1000):
    with open(file_path, "r", encoding="utf-8") as f:
        batch = []
        for line in f:
            batch.append(line.strip())  # Lägg till rad i batchen
            if len(batch) >= batch_size:
                yield batch  # Returnera en batch när den är full
                batch = []  # Rensa batchen för nästa
        if batch:
            yield batch  # Returnera den sista batchen om den inte är tom

def tokenize_function(examples):
    tokens = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()  # Lägg till labels för loss-beräkning
    return tokens

# Filväg för bearbetad data
DATA_FILE = "more_training_data.txt"

# Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 har ingen standard pad-token, så vi sätter eos som padding

# Skapa Hugging Face Dataset
train_dataset = Dataset.from_dict({"text": []})  # Initiera tom dataset först

# Ladda dataset i batcher och tokenisera
for batch in load_and_process_data_in_batches(DATA_FILE):
    # Skapa batch-data för dataset
    batch_dataset = Dataset.from_dict({"text": batch}).map(tokenize_function, batched=True)
    # Använd concatenate_datasets för att kombinera dataset
    if not train_dataset:
        train_dataset = batch_dataset  # Om det är första batchen, sätt den som train_dataset
    else:
        train_dataset = DatasetDict({"train": train_dataset["train"] + batch_dataset["train"]})  # Lägg till batchen i datasetet

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
    logging_dir="../logs",
    save_total_limit=3,  # Begränsa antalet sparade checkpoints
    save_steps=500  # Spara checkpoints var 500:e steg
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

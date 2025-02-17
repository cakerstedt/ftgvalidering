from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Ladda och förbered data från den bearbetade filen
def load_and_process_data(file_path):
    df = pd.read_csv(file_path)  # Läser in den bearbetade filen
    texts = df["verksamhetsbeskrivning"].tolist()  # Verksamhetsbeskrivningarna
    labels = df["sni_kod"].tolist()  # SNI-koderna som etiketter
    return texts, labels

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Filväg för bearbetad data
DATA_FILE = "processed_data.csv"

# Ladda dataset
train_texts, train_labels = load_and_process_data(DATA_FILE)

# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Omvandla labels till numerisk form
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)

# Skapa Hugging Face Dataset
train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels}).map(tokenize_function, batched=True)

# Initiera modell
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(set(train_labels)))

# Träningsargument
training_args = TrainingArguments(
    output_dir="../models/distilbert_model",
    evaluation_strategy="epoch",
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

# Spara modellen
model.save_pretrained("../models/distilbert_model")
tokenizer.save_pretrained("../models/distilbert_model")

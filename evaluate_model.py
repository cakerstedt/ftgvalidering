from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
from sklearn.metrics import accuracy_score

# Ladda modell
model = DistilBertForSequenceClassification.from_pretrained("../models/distilbert_model")
tokenizer = DistilBertTokenizer.from_pretrained("../models/distilbert_model")

def predict(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.argmax(logits, dim=-1).tolist()

# Testa modellen
test_preds = predict(test_texts)
accuracy = accuracy_score(test_labels, test_preds)
print(f"Test accuracy: {accuracy:.2f}")

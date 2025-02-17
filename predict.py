from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

# Ladda modell
model = DistilBertForSequenceClassification.from_pretrained("../models/distilbert_model")
tokenizer = DistilBertTokenizer.from_pretrained("../models/distilbert_model")

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    return predicted_class

# Exempel på företagsbeskrivning
description = "Vi säljer bilar och tillhandahåller fordonsservice."
predicted_category = predict(description)
print(f"Predicerad branschkod: {predicted_category}")

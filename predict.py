import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

path = "./harassment_model"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForSequenceClassification.from_pretrained(path)
model.eval()

def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    pred = int(np.argmax(probs))
    return {"label": pred, "prob_not": probs[0], "prob_harassment": probs[1]}

print(predict("she chatted me offensive messages"))
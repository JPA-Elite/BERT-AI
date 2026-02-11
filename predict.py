import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from data.test_data_prediction import test_data

path = "./harassment_model"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForSequenceClassification.from_pretrained(path)
model.eval()

LABEL_MAP = {
    0: "non_harassment",
    1: "harassment"
}

def predict(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()

    pred = int(np.argmax(probs))

    return {
        "pred_label": LABEL_MAP[pred],
        "prob_non_harassment": probs[0],
        "prob_harassment": probs[1]
    }

# -----------------------------
# RUN TESTS
# -----------------------------
correct = 0
threshold = 75  # percent confidence

print("\n--- MODEL TEST RESULTS ---\n")

for text, expected in test_data:
    result = predict(text)

    pred = result["pred_label"]
    confidence = (
        result["prob_harassment"] if pred == "harassment"
        else result["prob_non_harassment"]
    ) * 100

    is_correct = pred == expected
    if is_correct:
        correct += 1

    if not is_correct:
        print(f"Text: {text}")
        print(f"Expected: {expected}")
        print(f"Predicted: {pred}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"Result: {'✅ CORRECT' if is_correct else '❌ WRONG'}")
        print("-" * 50)

accuracy = (correct / len(test_data)) * 100
print(f"\nTOTAL ACCURACY: {accuracy:.2f}% ({correct}/{len(test_data)})\n")
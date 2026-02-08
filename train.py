import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random
from data.harassment import harassment_texts
from data.non_harassment import non_harassment_texts


harassment_texts = harassment_texts[:len(harassment_texts)]
non_harassment_texts = non_harassment_texts[:len(non_harassment_texts)]

texts = harassment_texts + non_harassment_texts
labels = [1] * len(harassment_texts) + [0] * len(non_harassment_texts)

# Shuffle dataset
combined = list(zip(texts, labels))
random.shuffle(combined)
texts, labels = zip(*combined)

ds = Dataset.from_dict({"text": list(texts), "label": list(labels)})

# -------------------------------
# Tokenization + Train
# -------------------------------

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True)

ds = ds.map(tokenize, batched=True)
ds = ds.train_test_split(test_size=0.2, seed=42)

ds["train"].set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
ds["test"].set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

args = TrainingArguments(
    output_dir="./harassment_model",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=20,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("./harassment_model")
tokenizer.save_pretrained("./harassment_model")

print("Training finished and model saved to ./harassment_model")
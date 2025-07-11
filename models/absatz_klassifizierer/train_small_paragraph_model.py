from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
import evaluate
import json
import torch
import nltk
from nltk.tokenize import sent_tokenize

# 1. Setup
print("CUDA available:", torch.cuda.is_available())
nltk.download("punkt")
nltk.download("punkt_tab")


def split_into_paragraphs(text, max_sentences=4):
    sentences = sent_tokenize(text)
    paragraphs = []
    for i in range(0, len(sentences), max_sentences):
        para = " ".join(sentences[i : i + max_sentences]).strip()
        if para:
            paragraphs.append(para)
    return paragraphs


def load_paragraph_data(path):
    paragraph_samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            text = sample["text"]
            label = sample["label"]
            paragraphs = split_into_paragraphs(text)
            for paragraph in paragraphs:
                paragraph_samples.append({"text": paragraph, "label": label})
    return paragraph_samples


# 2. Daten laden und reduzieren
samples = load_paragraph_data("../../data/dataset.jsonl")
samples = samples[:50000]

# 3. Split
train_data, temp_data = train_test_split(samples, train_size=40000, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=5000, random_state=42)

dataset = DatasetDict(
    {
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
        "test": Dataset.from_list(test_data),
    }
)

print("Train size:", len(dataset["train"]))
print("Validation size:", len(dataset["validation"]))
print("Test size:", len(dataset["test"]))

# 4. Modell & Tokenizer
model_name = "Hello-SimpleAI/chatgpt-detector-roberta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


# 5. Preprocessing
def preprocess(example):
    tokens = tokenizer(
        example["text"], truncation=True, padding="max_length", max_length=256
    )
    tokens["label"] = example["label"]
    return tokens


train_tokenized = dataset["train"].map(preprocess)
val_tokenized = dataset["validation"].map(preprocess)

# 6. Trainingsargumente
training_args = TrainingArguments(
    output_dir="./paragraph_model_output",
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=1000,
    logging_dir="./logs",
    logging_steps=20,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    learning_rate=2e-5,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
    fp16=torch.cuda.is_available(),
)


# 7. Metriken
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0, 1]
    )
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    return {
        "accuracy": (preds == labels).mean(),
        "precision_human": precision[0],
        "recall_human": recall[0],
        "f1_human": f1[0],
        "precision_ai": precision[1],
        "recall_ai": recall[1],
        "f1_ai": f1[1],
        "confusion_matrix": cm.tolist(),
    }


# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 9. Training starten
trainer.train()

# 10. Speichern
model.save_pretrained("./small_model")
tokenizer.save_pretrained("./small_tokenizer")

# 11. Testauswertung
test_tokenized = dataset["test"].map(preprocess)
results = trainer.evaluate(test_tokenized)

print("Test results:")
print(results)

with open("small_model_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

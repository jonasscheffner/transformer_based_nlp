from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from collections import defaultdict
import numpy as np
import evaluate
import json
import torch
import nltk
from nltk.tokenize import sent_tokenize

print("CUDA available:", torch.cuda.is_available())
nltk.download("punkt")
nltk.download("punkt_tab")


def split_into_paragraphs(text, max_sentences=10):
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
            category = sample.get("category", "unknown")
            paragraphs = split_into_paragraphs(text)
            for paragraph in paragraphs:
                paragraph_samples.append(
                    {"text": paragraph, "label": label, "category": category}
                )
    return paragraph_samples


samples = load_paragraph_data("../../data/dataset_no_dupes.jsonl")
samples = samples[:30000]

train_data, temp_data = train_test_split(samples, train_size=24000, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=3000, random_state=42)

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


model_name = "Hello-SimpleAI/chatgpt-detector-roberta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


def preprocess(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    tokens["label"] = example["label"]
    tokens["category"] = example.get("category", "unknown")
    return tokens


train_tokenized = dataset["train"].map(preprocess)
val_tokenized = dataset["validation"].map(preprocess)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="./paragraph_model_10sent_output",
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=1000,
    logging_dir="./logs_10sent",
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


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained("./paragraph_model_10sent")
tokenizer.save_pretrained("./paragraph_tokenizer_10sent")

test_tokenized = dataset["test"].map(preprocess)
results = trainer.evaluate(test_tokenized)

print("Test-Ergebnisse (gesamt):")
print(results)

with open("metrics_10sent.json", "w") as f:
    json.dump(results, f, indent=2)


# Evaluation pro Kategorie
sources_dict = defaultdict(list)
for sample in dataset["test"]:
    sources_dict[sample["category"]].append(sample)

for category in sources_dict.keys():
    test_data_category = sources_dict[category]
    tokenized = Dataset.from_list(test_data_category).map(preprocess)
    results = trainer.evaluate(tokenized)

    print(f"Test-Ergebnisse für Kategorie '{category}':")
    print(results)

    filename = f"{category}_metrics_10sent.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

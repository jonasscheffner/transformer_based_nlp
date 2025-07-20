from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import numpy as np
import evaluate
import torch
import json
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import nltk
from nltk.tokenize import sent_tokenize
import os

nltk.download("punkt_tab")
print("Cuda available:", torch.cuda.is_available())

def load_sentence_context_data(path, context_size=2):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            sentences = sent_tokenize(example["text"])
            for i in range(len(sentences)):
                start = max(0, i - context_size)
                end = min(len(sentences), i + context_size + 1)
                context = sentences[start:i] + [f"[TARGET] {sentences[i]} [/TARGET]"] + sentences[i+1:end]
                context_text = " ".join(context)
                samples.append({"text": context_text, "label": example["label"], "category": example["category"]})
    return samples

print("Loading samples")
samples = load_sentence_context_data("../../data/ganzer_text/dataset_no_dupes.jsonl", context_size=2)

train_data, temp_data = train_test_split(samples, train_size=40000, random_state=42)
test_data, val_data = train_test_split(temp_data, test_size=6000, random_state=42)

test_data = test_data[:100000]

dataset = DatasetDict({
    "train": Dataset.from_list(train_data),
    "validation": Dataset.from_list(val_data),
    "test": Dataset.from_list(test_data),
})

print(dataset)

model_name = "Hello-SimpleAI/chatgpt-detector-roberta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

special_tokens_dict = {"additional_special_tokens": ["[TARGET]", "[/TARGET]"]}
tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

def preprocess(example):
    tokenized = tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
    tokenized["label"] = example["label"]
    return tokenized

train_tokenized = dataset["train"].map(preprocess, batched=False)
validation_tokenized = dataset["validation"].map(preprocess, batched=False)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=5000,
    logging_steps=50,
    logging_dir="./logs",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    learning_rate=2e-5,
    warmup_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    fp16=True,                              # falls GPU benutzt wird am besten anmachen
    report_to="none"
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average=None, labels=[0, 1])
    cm = confusion_matrix(labels, predictions, labels=[0, 1])

    return {
        "accuracy": (predictions == labels).mean(),
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
    eval_dataset=validation_tokenized,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

model.save_pretrained("./sentence_classification_model")
tokenizer.save_pretrained("./sentence_classification_tokenizer")


results = trainer.evaluate()
print("results:")
print(results)

with open("metrics_train.json", "w") as f:
    json.dump(results, f, indent=2)

print("Test-Ergebnisse:")
print(results)

with open("metrics_train.json", "w") as f:
    json.dump(results, f, indent=2)



test_tokenized = Dataset.from_list(test_data).map(preprocess)
results = trainer.evaluate(test_tokenized)

print("Test-Ergebnisse für den gesamten Testdatensatz:")
print(results)

with open("metrics_whole_test_dataset.json", "w") as f:
    json.dump(results, f, indent=2)


from collections import defaultdict

sources_dict = defaultdict(list)
for sample in dataset["test"]:
    sources_dict[sample["category"]].append(sample)


for category in sources_dict.keys():
    test_data_category = sources_dict[category]
    tokenized = Dataset.from_list(test_data_category).map(preprocess)
    results = trainer.evaluate(tokenized)

    print(f"Test Results for {category}:")
    print(results)

    filename = f"{category}_metrics.jsonl"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

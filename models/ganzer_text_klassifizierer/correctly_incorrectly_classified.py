from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import evaluate
import torch
import json
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

print("Cuda available: ", torch.cuda.is_available())


def getJsonAsArray(path):
    samples = []
    for line in open(path):
        sample = json.loads(line)
        samples.append(sample)
    return samples

samples = getJsonAsArray("../../data/ganzer_text/dataset_no_dupes.jsonl")

train_data, temp_data = train_test_split(samples, train_size=75000, random_state=42)

test_data, val_data = train_test_split(temp_data, test_size=25000, random_state=42)


train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)
test_dataset = Dataset.from_list(test_data)

dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

print(dataset)

tokenizer = AutoTokenizer.from_pretrained("whole_text_classification_tokenizer")
model = AutoModelForSequenceClassification.from_pretrained("whole_text_classification_model", num_labels=2)

def preprocess(example):
    tokenized = tokenizer(example["text"], truncation=True)
    tokenized["label"] = example["label"]
    return tokenized

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



training_args = TrainingArguments(
    output_dir="./results",
    save_strategy="no",           
    logging_steps=50,
    logging_dir="./logs",
    per_device_eval_batch_size=8,
    num_train_epochs=1,          
    report_to="none",
    fp16=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

test_tokenized = dataset["test"].map(preprocess, batched=False)
predictions = trainer.predict(test_tokenized)

predicted_labels = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids

correct = 0
total_wrong = 0

correctly_classified_file = open("correctly_classified.txt", "w", encoding="utf-8")
incorrectly_classified_file = open("incorrectly_classified.txt", "w", encoding="utf-8")

for i, (pred, true) in enumerate(zip(predicted_labels, true_labels)):

    if (i+1)%500 == 0:
        print(f"Correct at {i} samples: {correct} accuracy: {correct/i}")

    text = dataset["test"][i]["text"]
    if pred == true:
        correct += 1
        correctly_classified_file.write(f"{dataset['test'][i]}\n")
    else:
        total_wrong += 1
        incorrectly_classified_file.write(f"{dataset['test'][i]}\n")

print(f"Correct predicted:{correct}")
print(f"Total accuracy on test data: {correct/len(dataset['test'])}")

correctly_classified_file.close()
incorrectly_classified_file.close()
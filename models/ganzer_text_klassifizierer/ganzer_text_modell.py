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

train_data, temp_data = train_test_split(samples, train_size=20000, random_state=42)

test_data, val_data = train_test_split(temp_data, test_size=5000, random_state=42)


train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)
test_dataset = Dataset.from_list(test_data)

dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

print(dataset)

model_name = "Hello-SimpleAI/chatgpt-detector-roberta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def preprocess(example):
    tokenized = tokenizer(example["text"], truncation=True)
    tokenized["label"] = example["label"]
    return tokenized

train_tokenized = dataset["train"].map(preprocess, batched=False)
validation_tokenized = dataset["validation"].map(preprocess, batched=False)

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

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

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

accuracy = evaluate.load("accuracy")
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
    compute_metrics=compute_metrics,
    data_collator=data_collator
)


trainer.train()

model.save_pretrained("./whole_text_classification_model")
tokenizer.save_pretrained("./whole_text_classification_tokenizer")


results = trainer.evaluate()
print("results:")
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


#import json
#
#results = trainer.evaluate()
#
#with open("metrics_results2.json", "w") as f:
#    json.dump(results, f, indent=4)
#
#
#
#test_tokenized = dataset["test"].map(preprocess, batched=False)
#predictions = trainer.predict(test_tokenized)
#
#predicted_labels = np.argmax(predictions.predictions, axis=1)
#true_labels = predictions.label_ids
#
#correct = 0
#total_wrong = 0
#
#correctly_classified_file = open("correctly_classified.txt", "w", encoding="utf-8")
#incorrectly_classified_file = open("incorrectly_classified.txt", "w", encoding="utf-8")
#
#for i, (pred, true) in enumerate(zip(predicted_labels, true_labels)):
#
#    if (i+1)%500 == 0:
#        print(f"Correct at {i} samples: {correct} accuracy: {correct/i}")
#
#    text = dataset["test"][i]["text"]
#    if pred == true:
#        correct += 1
#        correctly_classified_file.write(f"Actual Label: {true}, Predicted Label: {pred}\n")
#        correctly_classified_file.write(f"Text: {text}\n")
#        correctly_classified_file.write("-" * 50 + "\n")
#    else:
#        total_wrong += 1
#        incorrectly_classified_file.write(f"Actual Label: {true}, Predicted Label: {pred}\n")
#        incorrectly_classified_file.write(f"Text: {text}\n")
#        incorrectly_classified_file.write("-" * 50 + "\n")
#
#incorrectly_classified_file.write(f"Total wrong: {total_wrong} from total samples: {len(dataset['test'])}\n")
#incorrectly_classified_file.write(f"Accuracy correct: {correct / len(dataset['test'])}\n")
#
#print(f"Correct predicted:{correct}")
#print(f"Total accuracy on test data: {correct/len(dataset['test'])}")
#
#correctly_classified_file.close()
#incorrectly_classified_file.close()
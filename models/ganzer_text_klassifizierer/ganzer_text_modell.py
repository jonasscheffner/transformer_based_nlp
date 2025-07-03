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
        samples.append({"text": sample["text"], "label": sample["label"]})
    return samples

samples = getJsonAsArray("../../data/ganzer_text/dataset.jsonl")

train_data, temp_data = train_test_split(samples, train_size=45000, random_state=42)

val_data, test_data = train_test_split(temp_data, test_size=5000, random_state=42)

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
    tokenized = tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)
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
    logging_steps=20,
    logging_dir="./logs",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
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
)


trainer.train()

model.save_pretrained("./whole_text_classification_model")
tokenizer.save_pretrained("./whole_text_classification_tokenizer")


results = trainer.evaluate()
print("results:")
print(results)

import json

results = trainer.evaluate()

with open("metrics_results2.json", "w") as f:
    json.dump(results, f, indent=4)



tokenizerSaved = AutoTokenizer.from_pretrained("./whole_text_classification_tokenizer")
modelSaved = AutoModelForSequenceClassification.from_pretrained("./whole_text_classification_model", num_labels=2)

correct = 0

correctly_classified_file = open("correctly_classified.txt", "w", encoding="utf-8")
incorrectly_classified_file = open("incorrectly_classified.txt", "w", encoding="utf-8")

for i, sample in enumerate(dataset["test"]):
    text = sample["text"]
    label = sample["label"]

    if (i+1)%500 == 0:
        print(f"Correct at {i} samples: {correct} accuracy: {correct/i}")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            model.to('cuda')

        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

        if prediction == label:
            correct = correct + 1
            correctly_classified_file.write(f"Actual Label: {label}, Predicted Label: {prediction}\n")
            correctly_classified_file.write(f"Text: {text}\n")
            correctly_classified_file.write("-" * 50 + "\n")
        else:
            incorrectly_classified_file.write(f"Actual Label: {label}, Predicted Label: {prediction}\n")
            incorrectly_classified_file.write(f"Text: {text}\n")
            incorrectly_classified_file.write("-" * 50 + "\n")

print(f"Correct predicted:{correct}")
print(f"Total accuracy on test data: {correct/len(dataset['test'])}")

correctly_classified_file.close()
incorrectly_classified_file.close()












#from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
#import numpy as np
#import evaluate
#import torch
#
#tokenizer = AutoTokenizer.from_pretrained("./ai_human_classifier_gpt2_absatz_tokenizer")
#model = AutoModelForSequenceClassification.from_pretrained("./ai_human_classifier_gpt2_absatz_model", num_labels=2)
#
#import torch
#
#while True:
#    user_input = input("Gib einen Text ein (oder 'exit' zum Beenden): ")
#    if user_input.lower() in ["exit", "quit", "q"]:
#        print("Beendet.")
#        break
#
#    # Tokenisierung
#    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
#
#    # Vorhersage
#    with torch.no_grad():
#        outputs = model(**inputs)
#        prediction = torch.argmax(outputs.logits, dim=1).item()
#
#    # Ausgabe
#    label = "AI" if prediction == 1 else "Mensch"
#    print("Label:", label)
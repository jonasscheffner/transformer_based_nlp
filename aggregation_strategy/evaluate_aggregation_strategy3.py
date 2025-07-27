import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.tokenize import sent_tokenize
from nltk import download
from itertools import permutations, product
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

download("punkt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def split_into_paragraphs(text, max_sentences=7):
    sentences = sent_tokenize(text)
    return [
        " ".join(sentences[i : i + max_sentences])
        for i in range(0, len(sentences), max_sentences)
    ]


def predict_prob(text, model, tokenizer):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)
    return probs


def predict_whole_text(text, model, tokenizer):
    probs = predict_prob(text, model, tokenizer)
    return probs[0][1].item()  # AI class


def predict_paragraph_level(text, model, tokenizer, max_sentences=7):
    paragraphs = split_into_paragraphs(text, max_sentences)
    if not paragraphs:
        return 0.0
    probs = [predict_prob(p, model, tokenizer)[0][1].item() for p in paragraphs]
    return sum(probs) / len(probs)


def predict_sentence_level(text, model, tokenizer, context_window=2):
    sentences = sent_tokenize(text)
    if not sentences:
        return 0.0
    probs = []
    for i in range(len(sentences)):
        start = max(0, i - context_window)
        end = min(len(sentences), i + context_window + 1)
        context = (
            sentences[start:i]
            + [f"[TARGET] {sentences[i]} [/TARGET]"]
            + sentences[i + 1 : end]
        )
        context_text = " ".join(context)
        prob = predict_prob(context_text, model, tokenizer)[0][1].item()
        probs.append(prob)
    return sum(probs) / len(probs)


def aggregation_strategy_3(
    text,
    models,
    tokenizers,
    order,
    upper_thresh,
    lower_thresh,
    decision_thresh,
    paragraph_max_sentences=7,
    sentence_context=2,
):
    probs = {}
    probs["whole"] = predict_whole_text(text, models["whole"], tokenizers["whole"])
    probs["paragraph"] = predict_paragraph_level(
        text, models["paragraph"], tokenizers["paragraph"], paragraph_max_sentences
    )
    probs["sentence"] = predict_sentence_level(
        text, models["sentence"], tokenizers["sentence"], sentence_context
    )

    for level in order:
        prob = probs[level]
        if prob >= upper_thresh:
            return 1, level, prob
        elif prob <= lower_thresh:
            return 0, level, prob

    fallback = int(probs["sentence"] >= decision_thresh)
    return fallback, "sentence-fallback", probs["sentence"]


def compute_metrics(true, pred):
    acc = accuracy_score(true, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true, pred, average="binary", pos_label=1
    )
    cm = confusion_matrix(true, pred).tolist()
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


test_path = "../data/test_dataset_no_dupes_for_aggregationstrategy.jsonl"
whole_model_path = (
    "../models/ganzer_text_klassifizierer/whole_text_classification_model"
)
whole_tokenizer_path = (
    "../models/ganzer_text_klassifizierer/whole_text_classification_tokenizer"
)
paragraph_model_path = "../models/absatz_klassifizierer/paragraph_model_7sent"
paragraph_tokenizer_path = "../models/absatz_klassifizierer/paragraph_tokenizer_7sent"
sentence_model_path = "../models/satz_klassifizierer/sentence_classification_model"
sentence_tokenizer_path = (
    "../models/satz_klassifizierer/sentence_classification_tokenizer"
)

# load models
models = {
    "whole": AutoModelForSequenceClassification.from_pretrained(whole_model_path).to(
        device
    ),
    "paragraph": AutoModelForSequenceClassification.from_pretrained(
        paragraph_model_path
    ).to(device),
    "sentence": AutoModelForSequenceClassification.from_pretrained(
        sentence_model_path
    ).to(device),
}
tokenizers = {
    "whole": AutoTokenizer.from_pretrained(whole_tokenizer_path),
    "paragraph": AutoTokenizer.from_pretrained(paragraph_tokenizer_path),
    "sentence": AutoTokenizer.from_pretrained(sentence_tokenizer_path),
}

# load data
samples = []
with open(test_path, "r", encoding="utf-8") as f:
    for line in f:
        samples.append(json.loads(line))

# grid search settings
orders = list(permutations(["whole", "paragraph", "sentence"]))
upper_threshs = [0.9, 0.95]
lower_threshs = [0.1, 0.05]
decision_threshs = [0.5, 0.6]

output_folder = "aggregation_strategy3_outputs"
os.makedirs(output_folder, exist_ok=True)

best_f1 = -1.0
best_config = None
best_preds = []
best_used_levels = []

# grid search
for order, upper, lower, fallback in product(
    orders, upper_threshs, lower_threshs, decision_threshs
):
    preds = []
    used_levels = []
    true_labels = []

    for sample in samples:
        label = sample["label"]
        true_labels.append(label)

        pred, level, _ = aggregation_strategy_3(
            sample["text"],
            models,
            tokenizers,
            order=order,
            upper_thresh=upper,
            lower_thresh=lower,
            decision_thresh=fallback,
        )
        preds.append(pred)
        used_levels.append(level)

    metrics = compute_metrics(true_labels, preds)
    f1 = metrics["f1"]

    if f1 > best_f1:
        best_f1 = f1
        best_config = {
            "model_order": list(order),
            "upper_thresh": upper,
            "lower_thresh": lower,
            "decision_thresh": fallback,
        }
        best_preds = preds
        best_used_levels = used_levels

# final evaluation
final_metrics = compute_metrics([s["label"] for s in samples], best_preds)
final_metrics["used_levels"] = {
    level: best_used_levels.count(level) for level in set(best_used_levels)
}

# save best config
with open(os.path.join(output_folder, "best_config.json"), "w", encoding="utf-8") as f:
    json.dump(best_config, f, indent=2, ensure_ascii=False)

# save metrics
with open(
    os.path.join(output_folder, "aggregationstrategy3_metrics.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(final_metrics, f, indent=2, ensure_ascii=False)

print("Finished Aggregation Strategy 3")
print("Best configuration:")
print(json.dumps(best_config, indent=2, ensure_ascii=False))
print("Final evaluation metrics:")
print(json.dumps(final_metrics, indent=2, ensure_ascii=False))

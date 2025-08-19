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
    precomputed_probs, order, upper_thresh, lower_thresh, decision_thresh
):
    for level in order:
        prob = precomputed_probs[level]
        if prob >= upper_thresh:
            return 1, level, prob
        elif prob <= lower_thresh:
            return 0, level, prob
    fallback = int(precomputed_probs["sentence"] >= decision_thresh)
    return fallback, "sentence-fallback", precomputed_probs["sentence"]


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

# precompute probabilities for each sample and model
print("Precomputing model probabilities...")
precomputed = []
for i, sample in enumerate(samples, start=1):
    probs = {
        "whole": predict_whole_text(
            sample["text"], models["whole"], tokenizers["whole"]
        ),
        "paragraph": predict_paragraph_level(
            sample["text"], models["paragraph"], tokenizers["paragraph"]
        ),
        "sentence": predict_sentence_level(
            sample["text"], models["sentence"], tokenizers["sentence"]
        ),
    }
    precomputed.append(
        {"label": sample["label"], "text": sample["text"], "probs": probs}
    )
    if i % 5000 == 0:
        print(f"Precomputed {i}/{len(samples)} samples...")

# grid search settings
orders = list(permutations(["whole", "paragraph", "sentence"]))
upper_threshs = [0.9, 0.925, 0.95, 0.975]
lower_threshs = [0.025, 0.05, 0.075, 0.1]
decision_threshs = [0.5]

output_folder = "aggregation_strategy3_outputs"
os.makedirs(output_folder, exist_ok=True)

best_f1 = -1.0
best_config = None
best_preds = []
best_used_levels = []

combinations = list(product(orders, upper_threshs, lower_threshs, decision_threshs))
total_combinations = len(combinations)
print(f"Starting grid search with {total_combinations} combinations...")

# grid search
for idx, (order, upper, lower, fallback) in enumerate(combinations, start=1):
    print(f"\nRunning combination {idx}/{total_combinations}")
    preds = []
    used_levels = []
    true_labels = []

    for i, entry in enumerate(precomputed, start=1):
        label = entry["label"]
        true_labels.append(label)

        pred, level, _ = aggregation_strategy_3(
            entry["probs"],
            order=order,
            upper_thresh=upper,
            lower_thresh=lower,
            decision_thresh=fallback,
        )
        preds.append(pred)
        used_levels.append(level)

        if i % 5000 == 0:
            print(f"\nProcessed {i}/{len(precomputed)} samples for current config...")

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
final_metrics = compute_metrics([e["label"] for e in precomputed], best_preds)
final_metrics["used_levels"] = {
    level: best_used_levels.count(level) for level in set(best_used_levels)
}

print("Evaluating individual models...")
true_labels = [e["label"] for e in precomputed]

whole_preds = [int(e["probs"]["whole"] >= 0.5) for e in precomputed]
whole_metrics = compute_metrics(true_labels, whole_preds)

paragraph_preds = [int(e["probs"]["paragraph"] >= 0.5) for e in precomputed]
paragraph_metrics = compute_metrics(true_labels, paragraph_preds)

sentence_preds = [int(e["probs"]["sentence"] >= 0.5) for e in precomputed]
sentence_metrics = compute_metrics(true_labels, sentence_preds)

# aggregation correct while others wrong
correct_when_others_wrong = []
for i, entry in enumerate(precomputed):
    label = entry["label"]
    agg_pred = best_preds[i]
    whole_pred = whole_preds[i]
    para_pred = paragraph_preds[i]
    sent_pred = sentence_preds[i]

    if agg_pred == label and (
        whole_pred != label or para_pred != label or sent_pred != label
    ):
        correct_when_others_wrong.append(
            {
                "text": entry["text"],
                "label": label,
                "aggregation_pred": agg_pred,
                "whole_pred": whole_pred,
                "paragraph_pred": para_pred,
                "sentence_pred": sent_pred,
            }
        )

print(
    f"\nAggregation correct while at least one model was wrong: {len(correct_when_others_wrong)} cases"
)

# save metrics and config
all_metrics = {
    "aggregation_strategy_3": final_metrics,
    "individual_models": {
        "whole": whole_metrics,
        "paragraph": paragraph_metrics,
        "sentence": sentence_metrics,
    },
}

with open(
    os.path.join(output_folder, "aggregationstrategy3_metrics.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(all_metrics, f, indent=2, ensure_ascii=False)

with open(os.path.join(output_folder, "best_config.json"), "w", encoding="utf-8") as f:
    json.dump(best_config, f, indent=2, ensure_ascii=False)

with open(
    os.path.join(output_folder, "aggregation_correct_and_others_not.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(correct_when_others_wrong, f, indent=2, ensure_ascii=False)

print("Finished Aggregation Strategy 3")
print("Best configuration:")
print(json.dumps(best_config, indent=2, ensure_ascii=False))
print("Final metrics including individual models:")
print(json.dumps(all_metrics, indent=2, ensure_ascii=False))

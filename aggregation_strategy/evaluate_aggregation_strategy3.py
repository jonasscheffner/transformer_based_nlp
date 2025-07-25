import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.tokenize import sent_tokenize
from nltk import download
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
    paragraphs = split_into_paragraphs(text, max_sentences=max_sentences)
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
    whole_model,
    whole_tokenizer,
    paragraph_model,
    paragraph_tokenizer,
    sentence_model,
    sentence_tokenizer,
    paragraph_max_sentences=7,
    sentence_context=2,
    upper_thresh=0.95,
    lower_thresh=0.05,
    decision_thresh=0.5,
):
    prob_whole = predict_whole_text(text, whole_model, whole_tokenizer)
    if prob_whole >= upper_thresh:
        return 1, "whole", prob_whole
    elif prob_whole <= lower_thresh:
        return 0, "whole", prob_whole

    prob_paragraph = predict_paragraph_level(
        text,
        paragraph_model,
        paragraph_tokenizer,
        max_sentences=paragraph_max_sentences,
    )
    if prob_paragraph >= upper_thresh:
        return 1, "paragraph", prob_paragraph
    elif prob_paragraph <= lower_thresh:
        return 0, "paragraph", prob_paragraph

    prob_sentence = predict_sentence_level(
        text, sentence_model, sentence_tokenizer, context_window=sentence_context
    )
    if prob_sentence >= upper_thresh:
        return 1, "sentence", prob_sentence
    elif prob_sentence <= lower_thresh:
        return 0, "sentence", prob_sentence

    fallback = int(prob_sentence >= decision_thresh)
    return fallback, "sentence-fallback", prob_sentence


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

whole_model = AutoModelForSequenceClassification.from_pretrained(whole_model_path).to(
    device
)
whole_tokenizer = AutoTokenizer.from_pretrained(whole_tokenizer_path)
paragraph_model = AutoModelForSequenceClassification.from_pretrained(
    paragraph_model_path
).to(device)
paragraph_tokenizer = AutoTokenizer.from_pretrained(paragraph_tokenizer_path)
sentence_model = AutoModelForSequenceClassification.from_pretrained(
    sentence_model_path
).to(device)
sentence_tokenizer = AutoTokenizer.from_pretrained(sentence_tokenizer_path)


samples = []
with open(test_path, "r", encoding="utf-8") as f:
    for line in f:
        samples.append(json.loads(line))

output_folder = "aggregation_strategy3_outputs"
os.makedirs(output_folder, exist_ok=True)

true_labels = []
agg_preds = []
used_levels = []

for i, sample in enumerate(samples):
    if i % 50 == 0:
        print(f"{i}/{len(samples)} processed...")

    label = sample["label"]
    true_labels.append(label)

    agg_pred, level, prob = aggregation_strategy_3(
        sample["text"],
        whole_model,
        whole_tokenizer,
        paragraph_model,
        paragraph_tokenizer,
        sentence_model,
        sentence_tokenizer,
    )
    agg_preds.append(agg_pred)
    used_levels.append(level)

metrics = {
    "aggregation_strategy_3": compute_metrics(true_labels, agg_preds),
}
metrics["aggregation_strategy_3"]["used_levels"] = {
    level: used_levels.count(level) for level in set(used_levels)
}

with open(
    os.path.join(output_folder, "aggregationstrategy3_metrics.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print("Finished Aggregation Strategy 3")
print("Metrics:")
print(json.dumps(metrics, indent=2, ensure_ascii=False))

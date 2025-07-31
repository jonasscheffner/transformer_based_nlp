from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import torch
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

def split_into_paragraphs(text, max_sentences=7):
    sentences = sent_tokenize(text)
    paragraphs = []
    for i in range(0, len(sentences), max_sentences):
        para = " ".join(sentences[i: i + max_sentences]).strip()
        if para:
            paragraphs.append(para)
    return paragraphs


def predict_whole_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        human_prob = probabilities[0][0].item()
        ai_prob = probabilities[0][1].item()

    return {
        "prediction": predicted_class,
        "confidence": max(human_prob, ai_prob),
        "probabilities": {
            "human": human_prob,
            "ai": ai_prob
        }
    }


def predict_paragraph(text, tokenizer, model, max_sentences=7):
    paragraphs = split_into_paragraphs(text, max_sentences)

    model.eval()
    ai_probs = []
    predictions = []

    for paragraph in paragraphs:
        inputs = tokenizer(paragraph, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            ai_probs.append(probs[0][1].item())
            pred = torch.argmax(probs, dim=-1).item()
            predictions.append(pred)

    majority = int(sum(predictions) > len(predictions) / 2)
    avg_ai_prob = np.mean(ai_probs) if ai_probs else 0.5

    return {
        "prediction": majority,
        "paragraph_predictions": predictions,
        "num_paragraphs": len(paragraphs),
        "avg_ai_prob": avg_ai_prob
    }


def predict_sentence(text, tokenizer, model, context_size=2):
    sentences = sent_tokenize(text)

    model.eval()
    ai_probs = []
    predictions = []

    for i in range(len(sentences)):
        start = max(0, i - context_size)
        end = min(len(sentences), i + context_size + 1)
        context = sentences[start:i] + [f"[TARGET] {sentences[i]} [/TARGET]"] + sentences[i + 1:end]
        input_text = " ".join(context)

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            ai_probs.append(probs[0][1].item())
            pred = torch.argmax(probs, dim=-1).item()
            predictions.append(pred)

    majority = int(sum(predictions) > len(predictions) / 2)
    avg_ai_prob = np.mean(ai_probs) if ai_probs else 0.5

    return {
        "prediction": majority,
        "sentence_predictions": predictions,
        "num_sentences": len(sentences),
        "avg_ai_prob": avg_ai_prob
    }


def get_predictions(input, whole_text_classification_model, whole_text_classification_tokenizer,
                    paragraph_classification_model, paragraph_classification_tokenizer,
                    sentence_classification_model, sentence_classification_tokenizer):

    pred_whole_text = predict_whole_text(input, whole_text_classification_tokenizer, whole_text_classification_model)
    pred_paragraph = predict_paragraph(input, paragraph_classification_tokenizer, paragraph_classification_model)
    pred_sentence = predict_sentence(input, sentence_classification_tokenizer, sentence_classification_model)

    return pred_whole_text, pred_paragraph, pred_sentence


# Lade Modelle & Tokenizer
whole_text_tokenizer = AutoTokenizer.from_pretrained("../ganzer_text_modell_no_dupes_all_categories/whole_text_classification_tokenizer")
whole_text_model = AutoModelForSequenceClassification.from_pretrained("../ganzer_text_modell_no_dupes_all_categories/whole_text_classification_model", num_labels=2).to(device)

paragraph_tokenizer = AutoTokenizer.from_pretrained("../absatzweise/paragraph_tokenizer_7sent")
paragraph_model = AutoModelForSequenceClassification.from_pretrained("../absatzweise/paragraph_model_7sent", num_labels=2).to(device)

sentence_tokenizer = AutoTokenizer.from_pretrained("../satzweise_modell_no_dupes_all_categories/sentence_classification_tokenizer")
sentence_model = AutoModelForSequenceClassification.from_pretrained("../satzweise_modell_no_dupes_all_categories/sentence_classification_model", num_labels=2).to(device)


# Lade Daten
samples = []
with open("../../data/test_dataset_no_dupes_for_aggregationstrategy.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        samples.append(json.loads(line))

print("Anzahl Samples:", len(samples))

true_labels = []
whole_ai_probs = []
par_ai_probs = []
sen_ai_probs = []

for i, sample in enumerate(samples):
    if i % 50 == 0:
        print(f"Currently at {i} out of {len(samples)}")

    true_labels.append(sample["label"])

    whole_pred, par_pred, sen_pred = get_predictions(
        sample["text"],
        whole_text_classification_model=whole_text_model,
        whole_text_classification_tokenizer=whole_text_tokenizer,
        paragraph_classification_model=paragraph_model,
        paragraph_classification_tokenizer=paragraph_tokenizer,
        sentence_classification_model=sentence_model,
        sentence_classification_tokenizer=sentence_tokenizer
    )

    whole_ai_probs.append(whole_pred["probabilities"]["ai"])
    par_ai_probs.append(par_pred["avg_ai_prob"])
    sen_ai_probs.append(sen_pred["avg_ai_prob"])


X = np.array(list(zip(whole_ai_probs, par_ai_probs, sen_ai_probs)))
y = np.array(true_labels)

logreg = LogisticRegression()
logreg.fit(X, y)

weights = logreg.coef_[0]
bias = logreg.intercept_[0]

print("Gelerntes Gewicht whole_prob:", weights[0])
print("Gelerntes Gewicht par_prob:", weights[1])
print("Gelerntes Gewicht sen_prob:", weights[2])
print("Gelerntes Bias (Intercept):", bias)


weights_dict = {
    "weights": {
        "whole_prob": weights[0],
        "par_prob": weights[1],
        "sen_prob": weights[2],
    },
    "bias": bias
}

with open("logreg_weights_probs.json", "w", encoding="utf-8") as f:
    json.dump(weights_dict, f, indent=2)


aggregated_preds = logreg.predict(X)

accuracy = accuracy_score(y, aggregated_preds)
precision, recall, f1, _ = precision_recall_fscore_support(y, aggregated_preds, average='binary')
conf_matrix = confusion_matrix(y, aggregated_preds)

print("\n--- Aggregationsstrategie (LogReg auf Wahrscheinlichkeiten) ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)


wrong_whole = 0
wrong_paragraph = 0
wrong_sentence = 0
rescued_by_strategy = 0


whole_preds = []
par_preds = []
sen_preds = []

for i in range(len(samples)):
    true = y[i]
    pred = aggregated_preds[i]

    whole_label = int(whole_ai_probs[i] > 0.5)
    par_label = int(par_ai_probs[i] > 0.5)
    sen_label = int(sen_ai_probs[i] > 0.5)

    whole_preds.append(whole_label)
    par_preds.append(par_label)
    sen_preds.append(sen_label)

    if whole_label != true:
        wrong_whole += 1
    if par_label != true:
        wrong_paragraph += 1
    if sen_label != true:
        wrong_sentence += 1

    if pred == true and (
        whole_label != true or par_label != true or sen_label != true
    ):
        rescued_by_strategy += 1


print("\n--- Fehleranalyse ---")
print(f"Falschklassifikationen:")
print(f"Whole-Text-Modell:  {wrong_whole} von {len(samples)}")
print(f"Paragraph-Modell:   {wrong_paragraph} von {len(samples)}")
print(f"Sentence-Modell:    {wrong_sentence} von {len(samples)}")
print(f"Gerettet durch Strategie (LogReg korrekt, Einzelmodell(e) falsch): {rescued_by_strategy}")


metrics_whole = compute_metrics(y, whole_preds)
metrics_par = compute_metrics(y, par_preds)
metrics_sen = compute_metrics(y, sen_preds)


all_metrics = {"aggregated_strategy": {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1
}, "individual_models": {
    "whole_text_model": metrics_whole,
    "paragraph_model": metrics_par,
    "sentence_model": metrics_sen
}, "confusion_matrix_aggregated": conf_matrix.tolist(), "error_analysis": {
    "wrong_classifications": {
        "whole_text_model": wrong_whole,
        "paragraph_model": wrong_paragraph,
        "sentence_model": wrong_sentence
    },
    "rescued_by_strategy": rescued_by_strategy,
    "total_samples": len(samples)
}}


with open("detailed_model_metrics.json", "w", encoding="utf-8") as f:
    json.dump(all_metrics, f, indent=2)
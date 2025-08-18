from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def split_into_paragraphs(text, max_sentences=7):
    sentences = sent_tokenize(text)
    paragraphs = []
    for i in range(0, len(sentences), max_sentences):
        para = " ".join(sentences[i : i + max_sentences]).strip()
        if para:
            paragraphs.append(para)
    return paragraphs


def predict_whole_text(text, tokenizer, model):
    """
    Predicts if a text is human (label = 0) or AI (label = 1)
    Parameters:
        text: text to predict
    Returns:
        dict: prediction and probabilities
    """
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

    result = {
        "prediction": predicted_class,
        "confidence": max(human_prob, ai_prob),
        "probabilities": {
            "human": human_prob,
            "ai": ai_prob
        }
    }

    return result


def predict_paragraph(text, tokenizer, model, max_sentences=7):
    paragraphs = split_into_paragraphs(text, max_sentences)

    model.eval()

    predictions = []
    for paragraph in paragraphs:
        inputs = tokenizer(paragraph, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            predictions.append(pred)

    majority = int(sum(predictions) > len(predictions) / 2)
    return {
        "prediction": majority,
        "paragraph_predictions": predictions,
        "num_paragraphs": len(paragraphs),
    }


def predict_sentence(text, tokenizer, model, context_size=2):
    sentences = sent_tokenize(text)

    model.eval()

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
            pred = torch.argmax(probs, dim=-1).item()
            predictions.append(pred)

    majority = int(sum(predictions) > len(predictions) / 2)
    return {
        "prediction": majority,
        "sentence_predictions": predictions,
        "num_sentences": len(sentences),
    }

def aggregation_strategy1_prediction(input, whole_text_classification_model, whole_text_classification_tokenizer,
                          paragraph_classification_model, paragraph_classification_tokenizer,
                          sentence_classification_model, sentence_classification_tokenizer
                         ):
    """
    Uses all three models to classify given input string.
    Decision is made by majority vote
        Parameters:
            - input_text: input text to be classified
            - whole_text_classification_model: model for whole text classification
            - whole_text_classification_tokenizer: tokenizer for whole text model
            - paragraph_classification_model: model for paragraph classification
            - paragraph_classification_tokenizer: tokenizer for paragraph model
            - sentence_classification_model: model for sentence classification
            - sentence_classification_tokenizer: tokenizer for sentence model
        Returns:
            - prediction: aggregated prediction of the input text (0 = human, 1 = AI) and individual predictions (whole, par, sen)
    """
    pred_whole_text = predict_whole_text(input, whole_text_classification_tokenizer, whole_text_classification_model)
    pred_paragraph = predict_paragraph(input, paragraph_classification_tokenizer, paragraph_classification_model)
    pred_sentence = predict_sentence(input, sentence_classification_tokenizer, sentence_classification_model)

    output = 1 if (pred_whole_text["prediction"] + pred_paragraph["prediction"] + pred_sentence["prediction"]) >= 2 else 0

    return output, pred_whole_text, pred_paragraph, pred_sentence

def compute_metrics(true, pred):
    acc = accuracy_score(true, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average="binary", pos_label=1)
    cm = confusion_matrix(true, pred).tolist()

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }


whole_text_tokenizer = AutoTokenizer.from_pretrained("../ganzer_text_modell_no_dupes_all_categories/whole_text_classification_tokenizer")
whole_text_model = AutoModelForSequenceClassification.from_pretrained("../ganzer_text_modell_no_dupes_all_categories/whole_text_classification_model", num_labels=2).to(device)

paragraph_tokenizer = AutoTokenizer.from_pretrained("../absatzweise/paragraph_tokenizer_7sent")
paragraph_model = AutoModelForSequenceClassification.from_pretrained("../absatzweise/paragraph_model_7sent", num_labels=2).to(device)

sentence_tokenizer = AutoTokenizer.from_pretrained("../satzweise_modell_no_dupes_all_categories/sentence_classification_tokenizer")
sentence_model = AutoModelForSequenceClassification.from_pretrained("../satzweise_modell_no_dupes_all_categories/sentence_classification_model", num_labels=2).to(device)



samples = []
with open("../../data/test_dataset_no_dupes_for_aggregationstrategy.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        samples.append(json.loads(line))

print(len(samples))

true_labels = []

whole_preds = []
par_preds = []
sen_preds = []
agg_preds = []

aggregation_correct_and_others_not = []

for i, sample in enumerate(samples):
    if i%50 == 0: print(f"Currently at {i} out of {len(samples)}")

    true_labels.append(sample["label"])

    agg_pred, whole_pred, par_pred, sen_pred = aggregation_strategy1_prediction(sample["text"],
                                                                                whole_text_classification_model=whole_text_model,
                                                                                whole_text_classification_tokenizer=whole_text_tokenizer,
                                                                                paragraph_classification_model=paragraph_model,
                                                                                paragraph_classification_tokenizer=paragraph_tokenizer,
                                                                                sentence_classification_model=sentence_model,
                                                                                sentence_classification_tokenizer=sentence_tokenizer)

    whole_preds.append(whole_pred["prediction"])
    par_preds.append(par_pred["prediction"])
    sen_preds.append(sen_pred["prediction"])
    agg_preds.append(agg_pred)

    if agg_pred == sample["label"] and (whole_pred["prediction"] != sample["label"] or par_pred["prediction"] != sample["label"] or sen_pred["prediction"] != sample["label"]):
        aggregation_correct_and_others_not.append({
            "text": sample["text"],
            "label": sample["label"],
            "whole_pred": whole_pred["prediction"],
            "paragraph_pred": par_pred["prediction"],
            "sentence_pred": sen_pred["prediction"],
            "aggregation_pred": agg_pred
        })


with open("aggregation_correct_and_others_not.json", "w", encoding="utf-8") as f:
    json.dump(aggregation_correct_and_others_not, f, indent=2, ensure_ascii=False)

metrics = {
    "whole_text": compute_metrics(true_labels, whole_preds),
    "paragraph": compute_metrics(true_labels, par_preds),
    "sentence": compute_metrics(true_labels, sen_preds),
    "aggregation_strategy_1": compute_metrics(true_labels, agg_preds),
}

with open("aggregationstrategy_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print(f"aggregation_correct_and_others_not: {len(aggregation_correct_and_others_not)} out of {len(samples)}")
print("metrics:")
print(metrics)

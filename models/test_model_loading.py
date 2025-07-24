from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import torch


def test_model(model_path, tokenizer_path, model_name):
    print(f"\n🧪 Teste Modell '{model_name}' ...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        inputs = tokenizer("Dies ist ein Test.", return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        print(f"✅ Erfolg: {model_name} erfolgreich geladen und getestet.")
    except Exception as e:
        print(f"❌ Fehler beim Laden von {model_name}:\n{e}")


MODELS = [
    {
        "name": "Absatzklassifizierer (7-Satz)",
        "model_path": "absatz_klassifizierer/paragraph_model_7sent",
        "tokenizer_path": "absatz_klassifizierer/paragraph_tokenizer_7sent",
    },
    {
        "name": "Ganztextklassifizierer",
        "model_path": "ganzer_text_klassifizierer/whole_text_classification_model",
        "tokenizer_path": "ganzer_text_klassifizierer/whole_text_classification_tokenizer",
    },
    {
        "name": "Satzklassifizierer",
        "model_path": "satz_klassifizierer/sentence_classification_model",
        "tokenizer_path": "satz_klassifizierer/sentence_classification_tokenizer",
    },
]

if __name__ == "__main__":
    for model_info in MODELS:
        test_model(
            model_info["model_path"], model_info["tokenizer_path"], model_info["name"]
        )

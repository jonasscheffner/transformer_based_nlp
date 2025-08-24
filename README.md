# Praktikum Transformer Based NLP

Anmerkung: Pfade im Code müssen ggf. angepasst werden je nachdem wo auf der Maschine lokal die datasets/Modelle liegen

Enthalten sind 3 Ordner:
- data: Der data Ordner enthält 2 Jupyter Notebooks.
    - fetch_data lädt das PrismAI-Dataset, verarbeitet und vorbereitet die Daten für das finetuning der einzelnen Modelle und speichert das finale dataset in einer Datei namens dataset_no_dupes.jsonl ab
    - fetch_test_data_for_aggregationstrategy erstellt ein weiteres dataset test_dataset_no_dupes_for_aggregationstrategy.jsonl, welches Testdaten enthält, die nicht für das Trainiere/Validieren der Modelle benutzt wurden, mit denen die Aggregationsstrategien getestet werden. Es umfasst ca. 53.000 Texte mit labels
- models: Enthält den Code für das fine tunen von HelloSimpleAI/chatgpt-detector-roberta
    - ganzer_text_klassifizierer: Das Modell, das textweise den ganzen Input betrachtet und vorhersagt, ob dieser menschlich oder KI generiert ist
        - whole_text_classification_model: Enthält das abgespeicherte fertige Modell
        - whole_text_classification_tokenizer: Analog den Tokenizer
        - metrics: Enthält metriken, wie gut das Modell bei jeder Kategorie aus dem PrismAI-Dataset abgeschnitten hat sowie für das gesamte dataset
        - ganzer_text_modell.py: Code zum finetunen des modells
        - correctly_incorrectly_classified.py: skript, was 2 dateien erstellt, welche enthalten, welche Samples das Modell richtig bzw. falsch klassifiziert hat
    - absatz_klassifizierer: analog zum ganzen_text_klassifizierer, nur dass das Modell auf Absätze fine tuned ist und nicht textweise.
    - satz_klassifizierer: analog wie die beiden oben drüber
- aggregation_strategy: Enthält code für die 3 Aggregationsstrategien, welche in Rahmen von diesem Projekt untersucht wurden
    - strategy_1_majority_vote: Lässt 3 Modelle abstimmen, die Mehrheit ist die finale Vorhersage
        - strategy1.py: Code für die Strategie, testet diese auf dem test dataset und speichert Metriken in einer jsonl Datei ab
        - aggregationstrategy_metrics.jsonl: Enthält die Metriken von strategy1.py
        aggregation_correct_and_others_not.jsonl: Enthält alle Texte, welche die Strategie richtig klassifiziert hat, mindestens eins der Modelle jedoch nicht (in anderen Worten wann die Strategie gerettet hat)
    - strategy_2_weighted: Analog aufgebaut wie strategy_1_majority_vote
        - Enthält den Code einmal für linear regression und einmal für DecisionTree
    - strategy_3_hierarchical_strategy: analog, enthält zusätzlich noch die beste config für de hierarchie
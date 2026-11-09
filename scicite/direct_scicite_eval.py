import json
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from allennlp.common.util import import_submodules
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

MODEL = r".\models\scicite.tar.gz"
TEST = r"C:\Users\mahdi\Downloads\scicite\scicite\test.jsonl"
OUT = r".\predictions_scicite_test.jsonl"

print("Importing SciCite...")
import_submodules("scicite")

print("Loading model...")
archive = load_archive(
    MODEL,
    overrides='{"model":{"data_format":""}}'
)

predictor = Predictor.from_archive(
    archive,
    "predictor_scicite"
)

print("Reading UTF-8 test data...")
with open(TEST, "r", encoding="utf-8") as f:
    examples = [json.loads(line) for line in f if line.strip()]

print("Examples:", len(examples))

gold = []
pred = []

BATCH_SIZE = 16

with open(OUT, "w", encoding="utf-8") as fout:
    for start in range(0, len(examples), BATCH_SIZE):
        batch = examples[start:start+BATCH_SIZE]

        results = predictor.predict_batch_json(batch)

        for example, result in zip(batch, results):
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            gold.append(example["label"])
            pred.append(result["prediction"])

        fout.flush()
        print("Completed:", min(start + BATCH_SIZE, len(examples)), "/", len(examples))

print("\nFINISHED")
print("Predictions:", len(pred))
print("Accuracy: {:.4f}".format(accuracy_score(gold, pred)))
print("Macro F1: {:.4f}".format(f1_score(gold, pred, average="macro")))

print("\nClassification report:")
print(classification_report(gold, pred, digits=4))

labels = ["background", "method", "result"]
print("\nConfusion matrix")
print("Labels:", labels)
print(confusion_matrix(gold, pred, labels=labels))

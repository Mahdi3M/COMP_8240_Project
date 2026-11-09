import json
import torch

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from allennlp.common.util import import_submodules
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

print("Importing SciCite...")
import_submodules("scicite")

from scicite.data import read_s2_excerpt
from scicite.helper import is_sentence

MODEL = r".\models\scicite.tar.gz"
TEST = r"C:\Users\mahdi\Downloads\scicite\scicite\test.jsonl"
OUT = r".\predictions_scicite_test.jsonl"

BATCH_SIZE = 16

print("Loading model...")
archive = load_archive(
    MODEL,
    overrides='{"model":{"data_format":""}}'
)

predictor = Predictor.from_archive(
    archive,
    "predictor_scicite"
)

model = predictor._model
reader = predictor._dataset_reader
model.eval()

print("Reading UTF-8 test set...")
with open(TEST, "r", encoding="utf-8") as f:
    examples = [json.loads(line) for line in f if line.strip()]

print("Examples:", len(examples))

gold = []
pred = []

with open(OUT, "w", encoding="utf-8") as fout:

    for start in range(0, len(examples), BATCH_SIZE):

        batch = examples[start:start + BATCH_SIZE]

        instances = []
        positions = []
        results = [None] * len(batch)

        for i, item in enumerate(batch):

            citation = read_s2_excerpt(item)

            # Same validity test used by the official SciCite predictor
            if len(citation.text) < 5 or not is_sentence(citation.text):

                results[i] = {
                    "citingPaperId": item.get("citingPaperId"),
                    "citedPaperId": item.get("citedPaperId"),
                    "prediction": ""
                }

            else:

                instance = reader.text_to_instance(
                    citation_text=citation.text,
                    intent=citation.intent,
                    citing_paper_id=citation.citing_paper_id,
                    cited_paper_id=citation.cited_paper_id,
                    citation_excerpt_index=citation.citation_excerpt_index
                )

                instances.append(instance)
                positions.append(i)

        if instances:
            with torch.no_grad():
                outputs = model.forward_on_instances(instances)

            for pos, output in zip(positions, outputs):
                results[pos] = {
                    "citingPaperId": output.get("citing_paper_id"),
                    "citedPaperId": output.get("cited_paper_id"),
                    "prediction": output["prediction"]
                }

        for item, result in zip(batch, results):

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

            gold.append(item["label"])
            pred.append(result["prediction"])

        fout.flush()

        print(
            "Completed:",
            min(start + BATCH_SIZE, len(examples)),
            "/",
            len(examples)
        )

print("\n=== FINISHED ===")
print("Predictions:", len(pred))
print("Accuracy: {:.4f}".format(accuracy_score(gold, pred)))
print("Macro F1: {:.4f}".format(
    f1_score(gold, pred, average="macro")
))

print("\nClassification report:")
print(classification_report(gold, pred, digits=4))

labels = ["background", "method", "result"]

print("\nConfusion matrix")
print("Labels:", labels)
print(confusion_matrix(gold, pred, labels=labels))

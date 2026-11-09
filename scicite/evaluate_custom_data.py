import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, f1_score

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "final_data" / "citations_to_annotate.csv"
LABELS = ["background", "method", "result"]

with SOURCE.open("r", encoding="utf-8-sig", newline="") as handle:
    all_rows = list(csv.DictReader(handle))

rows = []
seen_ids = set()

for row in all_rows:
    example_id = row["example_id"].strip()
    include = row["include"].strip().lower()

    if include not in ("yes", "no"):
        raise SystemExit("Invalid include value for " + example_id)
    if example_id in seen_ids:
        raise SystemExit("Duplicate example ID: " + example_id)
    seen_ids.add(example_id)

    if include == "no":
        continue

    label = row["human_label"].strip().lower()
    if label not in LABELS:
        raise SystemExit("Invalid or missing label for " + example_id)
    if not row["citation_context"].strip():
        raise SystemExit("Empty passage for " + example_id)

    row["normalized_label"] = label
    rows.append(row)

if not rows:
    raise SystemExit("No included examples found.")

run_dir = ROOT / "final_results" / (
    "custom_" + datetime.now().strftime("%Y%m%d_%H%M%S")
)
run_dir.mkdir(parents=True, exist_ok=False)

# Freeze the annotations used for this evaluation.
source_bytes = SOURCE.read_bytes()
(run_dir / "annotations_snapshot.csv").write_bytes(source_bytes)

input_path = run_dir / "model_input.jsonl"
prediction_path = run_dir / "predictions.jsonl"

with input_path.open("w", encoding="ascii") as handle:
    for index, row in enumerate(rows):
        record = {
            "string": row["citation_context"],
            "sectionName": row["section"],
            "citingPaperId": row["source_pmcid"],
            # A local alignment ID, not a real bibliographic paper ID.
            "citedPaperId": "local:" + row["example_id"],
            "excerpt_index": index,
            # The old reader requires a label field. Human gold labels
            # are withheld from inference and used only for scoring.
            "label": "background"
        }
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")

metadata = {
    "annotation_sha256": hashlib.sha256(source_bytes).hexdigest(),
    "included_examples": len(rows),
    "excluded_examples": len(all_rows) - len(rows),
    "label_counts": dict(Counter(
        row["normalized_label"] for row in rows
    )),
    "python_executable": sys.executable,
    "model": str(ROOT / "models" / "scicite.tar.gz"),
    "note": "Gold labels withheld from inference; local IDs used for alignment."
}
with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
    json.dump(metadata, handle, indent=2)

print("Examples:", len(rows), flush=True)
print("Results folder:", run_dir, flush=True)
print("Loading the original SciCite model...", flush=True)

command = [
    sys.executable, "-m", "allennlp.run", "predict",
    str(ROOT / "models" / "scicite.tar.gz"),
    str(input_path),
    "--predictor", "predictor_scicite",
    "--include-package", "scicite",
    "--overrides", json.dumps({"model": {"data_format": ""}}),
    "--output-file", str(prediction_path),
    "--silent"
]

completed = subprocess.run(command, cwd=str(ROOT))
if completed.returncode:
    raise SystemExit("Prediction failed. Send the terminal error for review.")

with prediction_path.open("r", encoding="utf-8") as handle:
    outputs = [json.loads(line) for line in handle if line.strip()]

if len(outputs) != len(rows):
    raise SystemExit("Prediction count does not match input count.")

for row, output in zip(rows, outputs):
    expected = (row["source_pmcid"], "local:" + row["example_id"])
    actual = (output.get("citingPaperId"), output.get("citedPaperId"))
    if expected != actual:
        raise SystemExit("Prediction alignment failed for " + row["example_id"])

gold = [row["normalized_label"] for row in rows]
pred = [output.get("prediction", "") for output in outputs]
invalid = sum(value not in LABELS for value in pred)

accuracy = accuracy_score(gold, pred)
macro_f1 = f1_score(gold, pred, labels=LABELS, average="macro")
baseline = accuracy_score(gold, ["background"] * len(gold))

report = classification_report(
    gold, pred, labels=LABELS, digits=4
)

summary = (
    "CUSTOM DATASET EVALUATION\n"
    "Examples: {}\n"
    "Invalid/empty predictions: {}\n"
    "Accuracy: {:.4f}\n"
    "Three-class macro-F1: {:.4f}\n"
    "Always-background accuracy: {:.4f}\n\n{}"
).format(len(rows), invalid, accuracy, macro_f1, baseline, report)

(run_dir / "metrics.txt").write_text(summary, encoding="utf-8")

details = []
matrix = {label: {column: 0 for column in LABELS + ["invalid"]}
          for label in LABELS}

for row, actual, predicted in zip(rows, gold, pred):
    column = predicted if predicted in LABELS else "invalid"
    matrix[actual][column] += 1
    details.append({
        "example_id": row["example_id"],
        "topic": row["topic"],
        "human_label": actual,
        "prediction": predicted,
        "correct": actual == predicted,
        "citation_context": row["citation_context"]
    })

with (run_dir / "results_by_example.json").open(
    "w", encoding="utf-8"
) as handle:
    json.dump(details, handle, ensure_ascii=False, indent=2)

with (run_dir / "confusion_matrix.json").open(
    "w", encoding="utf-8"
) as handle:
    json.dump(matrix, handle, indent=2)

print("\n" + summary)
print("\nResults saved in:", run_dir)

import hashlib
import html
import json
import random
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, classification_report, f1_score
)

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "final_data/multicite/test.json"
MAPPING = {"background": "background", "uses": "method"}
LABELS = ["background", "method"]
SEED = 8240
PER_CLASS = 100

source_bytes = SOURCE.read_bytes()
data = json.loads(source_bytes.decode("utf-8-sig"))

# Exclude exact text overlaps with the locally available SciCite test set.
# This is not a complete check against the original training data.
def normalize(text):
    return re.sub(r"\W+", " ", text.casefold()).strip()

original_test_texts = set()
original_test = ROOT / "test_scicite_safe.jsonl"
if original_test.exists():
    with original_test.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                original_test_texts.add(
                    normalize(json.loads(line)["string"])
                )

eligible = []
excluded = Counter()
seen_ids = set()

for item in data:
    label = item["y"].strip()

    if label not in MAPPING:
        excluded["unsupported_or_multiple_labels"] += 1
        continue

    if not isinstance(item["x"], list) or not all(
        isinstance(text, str) for text in item["x"]
    ):
        raise SystemExit("Unexpected context structure: " + item["id"])

    # Remove only target-marking tags; preserve their citation text.
    text = " ".join(item["x"])
    text = re.sub(r"</?cite\b[^>]*>", "", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        excluded["empty_context"] += 1
        continue

    if item["id"] in seen_ids:
        raise SystemExit("Duplicate source ID: " + item["id"])
    seen_ids.add(item["id"])

    if normalize(text) in original_test_texts:
        excluded["exact_normalized_overlap_with_scicite_test"] += 1
        continue

    eligible.append({
        "example_id": item["id"],
        "original_label": label,
        "human_label": MAPPING[label],
        "citation_context": text
    })

# Remove repeated normalized contexts, including potentially conflicting
# labels. This rule is applied before sampling or observing predictions.
text_counts = Counter(normalize(row["citation_context"]) for row in eligible)
unique = []
for row in eligible:
    if text_counts[normalize(row["citation_context"])] > 1:
        excluded["repeated_context_record"] += 1
    else:
        unique.append(row)

rng = random.Random(SEED)
selected = []

for label in LABELS:
    pool = sorted(
        [row for row in unique if row["human_label"] == label],
        key=lambda row: row["example_id"]
    )
    print("Eligible {} examples: {}".format(label, len(pool)))
    if len(pool) < PER_CLASS:
        raise SystemExit("Insufficient examples. Send this output for review.")
    selected.extend(rng.sample(pool, PER_CLASS))

rng.shuffle(selected)

run_dir = ROOT / "final_results" / (
    "multicite_" + datetime.now().strftime("%Y%m%d_%H%M%S")
)
run_dir.mkdir(parents=True, exist_ok=False)

(run_dir / "selected_examples.json").write_text(
    json.dumps(selected, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

manifest = {
    "source": "MultiCite classification_1_context/test.json",
    "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
    "source_records": len(data),
    "mapping": MAPPING,
    "excluded_counts": dict(excluded),
    "sampling_seed": SEED,
    "examples_per_class": PER_CLASS,
    "selected_examples": len(selected),
    "scoring": "Macro-F1 over background and method; all outputs retained.",
    "limitations": [
        "Adapted two-category subset, not the full MultiCite benchmark.",
        "Balanced sampling changes the original label distribution.",
        "Exact-text overlap checked only against locally available SciCite test.",
        "Training-data and paper-level overlap have not been ruled out."
    ]
}
(run_dir / "selection_manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)

input_path = run_dir / "model_input.jsonl"
output_path = run_dir / "predictions.jsonl"

with input_path.open("w", encoding="ascii") as handle:
    for index, row in enumerate(selected):
        record = {
            "string": row["citation_context"],
            "sectionName": "",
            # Local alignment identifiers, not real paper identifiers.
            "citingPaperId": "multicite:" + row["example_id"],
            "citedPaperId": "target:" + row["example_id"],
            "excerpt_index": index,
            # Required by the legacy reader; gold withheld from inference.
            "label": "background"
        }
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")

print("\nSelected {} examples.".format(len(selected)), flush=True)
print("Results folder:", run_dir, flush=True)
print("Running the original SciCite model. Please wait...", flush=True)

command = [
    sys.executable, "-m", "allennlp.run", "predict",
    str(ROOT / "models/scicite.tar.gz"),
    str(input_path),
    "--predictor", "predictor_scicite",
    "--include-package", "scicite",
    "--overrides", json.dumps({"model": {"data_format": ""}}),
    "--output-file", str(output_path),
    "--silent"
]

completed = subprocess.run(command, cwd=str(ROOT))
if completed.returncode:
    raise SystemExit("Prediction failed. Send the terminal error.")

with output_path.open("r", encoding="utf-8") as handle:
    outputs = [json.loads(line) for line in handle if line.strip()]

if len(outputs) != len(selected):
    raise SystemExit("Prediction count mismatch.")

for row, output in zip(selected, outputs):
    expected = (
        "multicite:" + row["example_id"],
        "target:" + row["example_id"]
    )
    actual = (
        output.get("citingPaperId"),
        output.get("citedPaperId")
    )
    if expected != actual:
        raise SystemExit("Alignment error: " + row["example_id"])

gold = [row["human_label"] for row in selected]
pred = [output.get("prediction", "") for output in outputs]

accuracy = accuracy_score(gold, pred)
macro_f1 = f1_score(gold, pred, labels=LABELS, average="macro")
report = classification_report(gold, pred, labels=LABELS, digits=4)
invalid = sum(p not in ["background", "method", "result"] for p in pred)

summary = (
    "MULTICITE ADAPTED TWO-CATEGORY EVALUATION\n"
    "Examples: {}\n"
    "Accuracy: {:.4f}\n"
    "Two-class macro-F1: {:.4f}\n"
    "Always-background accuracy: 0.5000\n"
    "Result predictions (counted as errors): {}\n"
    "Invalid/empty predictions (counted as errors): {}\n\n{}"
).format(
    len(selected), accuracy, macro_f1,
    pred.count("result"), invalid, report
)

(run_dir / "metrics.txt").write_text(summary, encoding="utf-8")

details = []
matrix = {
    label: {p: 0 for p in ["background", "method", "result", "invalid"]}
    for label in LABELS
}

for row, prediction in zip(selected, pred):
    detail = dict(row)
    detail["prediction"] = prediction
    detail["correct"] = prediction == row["human_label"]
    details.append(detail)
    column = prediction if prediction in [
        "background", "method", "result"
    ] else "invalid"
    matrix[row["human_label"]][column] += 1

(run_dir / "results_by_example.json").write_text(
    json.dumps(details, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
(run_dir / "confusion_matrix.json").write_text(
    json.dumps(matrix, indent=2), encoding="utf-8"
)

print("\n" + summary)
print("\nResults saved in:", run_dir)

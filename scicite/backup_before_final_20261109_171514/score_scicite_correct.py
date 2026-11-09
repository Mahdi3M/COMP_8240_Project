import json
from sklearn.metrics import accuracy_score, f1_score, classification_report

TEST = r"C:\Users\mahdi\Downloads\scicite\scicite\test.jsonl"
PRED = r".\predictions_scicite_test.jsonl"

labels = ["background", "method", "result"]

with open(TEST, "r", encoding="utf-8") as f:
    gold_data = [json.loads(x) for x in f if x.strip()]

with open(PRED, "r", encoding="utf-8") as f:
    pred_data = [json.loads(x) for x in f if x.strip()]

gold = [x["label"] for x in gold_data]
pred = [x["prediction"] for x in pred_data]

invalid = [
    (i, gold[i], pred[i])
    for i in range(len(pred))
    if pred[i] not in labels
]

print("Examples:", len(gold))
print("Predictions:", len(pred))
print("Empty/invalid predictions:", len(invalid))
print("Accuracy:", "{:.4f}".format(accuracy_score(gold, pred)))

print(
    "3-CLASS MACRO F1:",
    "{:.4f}".format(
        f1_score(
            gold,
            pred,
            labels=labels,
            average="macro"
        )
    )
)

print("\n3-class report:")
print(
    classification_report(
        gold,
        pred,
        labels=labels,
        digits=4
    )
)

print("\nInvalid prediction indices:")
for x in invalid:
    print(x)

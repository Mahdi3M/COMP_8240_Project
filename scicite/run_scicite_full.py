import subprocess
import sys

cmd = [
    sys.executable,
    "-m", "allennlp.run",
    "predict",
    r".\models\scicite.tar.gz",
    r".\test_scicite_safe.jsonl",
    "--predictor", "predictor_scicite",
    "--include-package", "scicite",
    "--overrides", '{"model":{"data_format":""}}',
    "--output-file", r".\predictions_scicite_test.jsonl",
    "--batch-size", "16",
    "--silent"
]

print("Running full SciCite test set...")
result = subprocess.run(cmd)
sys.exit(result.returncode)

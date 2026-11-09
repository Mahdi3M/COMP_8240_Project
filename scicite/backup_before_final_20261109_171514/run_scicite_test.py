import subprocess
import sys

cmd = [
    sys.executable,
    "-m", "allennlp.run",
    "predict",
    r".\models\scicite.tar.gz",
    r".\test5.jsonl",
    "--predictor", "predictor_scicite",
    "--include-package", "scicite",
    "--overrides", '{"model":{"data_format":""}}',
    "--output-file", r".\predictions_test5.jsonl"
]

print("Running command:")
print(cmd)

result = subprocess.run(cmd)
sys.exit(result.returncode)

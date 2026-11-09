import json
from pathlib import Path

src = Path(r"C:\Users\mahdi\Downloads\scicite\scicite\test.jsonl")
dst = Path(r".\test_scicite_safe.jsonl")

count = 0

with src.open("r", encoding="utf-8") as fin, \
     dst.open("w", encoding="ascii", newline="\n") as fout:

    for line_no, line in enumerate(fin, 1):
        if not line.strip():
            continue

        obj = json.loads(line)

        # ensure_ascii=True converts non-ASCII characters to \uXXXX,
        # so old AllenNLP on Windows can safely read the file.
        fout.write(json.dumps(obj, ensure_ascii=True) + "\n")
        count += 1

print("Created:", dst)
print("Records:", count)

import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "eval/questions_hard.jsonl"
kept, removed = [], []

for n, line in enumerate(open(path), start=1):
    if not line.strip():
        continue
    try:
        q = json.loads(line)
    except json.JSONDecodeError:
        removed.append(f"line {n}: broken JSON")
        continue
    ref = q.get("reference", "")
    if "Passage A" in ref or "Passage B" in ref:
        removed.append(f"line {n}: reference mentions passages -> {q['question'][:60]}")
        continue
    kept.append(q)

with open(path, "w") as f:
    for q in kept:
        f.write(json.dumps(q) + "\n")

print("Removed:")
for r in removed:
    print("  " + r)
print(f"Kept {len(kept)} questions in {path}")

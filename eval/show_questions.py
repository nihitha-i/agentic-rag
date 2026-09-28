import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "eval/questions_hard.jsonl"
for i, line in enumerate(open(path), start=1):
    q = json.loads(line)
    print(f"{i}. Q: {q['question']}")
    print(f"   A: {q['reference']}")
    print(f"   ({q['source']})\n")
    
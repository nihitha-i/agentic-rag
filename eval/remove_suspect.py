import json

PATH = "eval/questions_hard.jsonl"

# Questions whose reference answer (answer key) looked unreliable
SUSPECT = [
    "hospital stay that wasn't covered",
    "doesn't get the doctor's approval",
    "considered self-administered",
    "also bill the VA",
    "heart condition",
    "needs dialysis",
    "partial hospitalization program",
    "occupational therapist",
]

kept, removed = [], []
for line in open(PATH):
    if not line.strip():
        continue
    q = json.loads(line)
    if any(s.lower() in q["question"].lower() for s in SUSPECT):
        removed.append(q["question"])
    else:
        kept.append(q)

with open(PATH, "w") as f:
    for q in kept:
        f.write(json.dumps(q) + "\n")

print(f"Removed {len(removed)}:")
for r in removed:
    print("  - " + r)
print(f"Kept {len(kept)} questions")

import sys

import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "results/questions_hard_details.csv"
system = sys.argv[2] if len(sys.argv) > 2 else "agent"

df = pd.read_csv(path)
df = df[df.system == system]

print(f"=== {system}: TRICK questions it did NOT refuse ===\n")
for _, r in df[(~df.answerable) & (~df.correct)].iterrows():
    print(f"Q: {r.question}\nA: {r.answer}\n")

print(f"=== {system}: REAL questions marked wrong or incomplete ===\n")
for _, r in df[df.answerable & (~df.correct | ~df.complete)].iterrows():
    flags = [name for name in ("correct", "complete", "faithful") if not r[name]]
    print(f"Q: {r.question}\n   failed: {', '.join(flags)}\n   A: {r.answer[:400]}\n")
import json
import sys
import time
from pathlib import Path

import pandas as pd

from app.baseline import answer as baseline_answer
from app.graph import ask as agent_answer
from app.llm import chat

JUDGE_MODEL = "gpt-4o"
JUDGE = (
    "You are a strict grader for a RAG system that answers Medicare policy questions. "
    "Reply in JSON with three booleans: "
    '{"correct": the answer agrees with the REFERENCE on the key facts '
    '(a correct "I don\'t know" for an unanswerable question counts as correct), '
    '"complete": the answer includes ALL important exceptions or conditions in the REFERENCE, '
    '"faithful": every claim in the answer is supported by the CONTEXT}.'
)


def judge(q: dict, result: dict) -> dict:
    out = chat(
        JUDGE,
        f"QUESTION: {q['question']}\nREFERENCE: {q['reference']}\n\n"
        f"CONTEXT:\n{chr(10).join(result['contexts'])[:12000]}\n\nANSWER:\n{result['answer']}",
        json_mode=True,
        model=JUDGE_MODEL,
    )
    return json.loads(out)


def main():
    qfile = sys.argv[1] if len(sys.argv) > 1 else "eval/questions.jsonl"
    name = Path(qfile).stem
    questions = [json.loads(line) for line in open(qfile) if line.strip()]
    rows = []
    for i, q in enumerate(questions, start=1):
        for system, fn in (("baseline", baseline_answer), ("agent", agent_answer)):
            start = time.time()
            result = fn(q["question"])
            seconds = time.time() - start
            grade = judge(q, result)
            rows.append({
                "system": system,
                "question": q["question"],
                "answerable": q["source"] is not None,
                "correct": bool(grade.get("correct")),
                "complete": bool(grade.get("complete")),
                "faithful": bool(grade.get("faithful")),
                "latency_s": round(seconds, 2),
                "answer": result["answer"],
            })
        print(f"{i}/{len(questions)} done")

    df = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    df.to_csv(f"results/{name}_details.csv", index=False)

    summary = df.groupby("system").agg(
        accuracy=("correct", "mean"),
        completeness=("complete", "mean"),
        faithfulness=("faithful", "mean"),
        avg_latency_s=("latency_s", "mean"),
    )
    for col in ("accuracy", "completeness", "faithfulness"):
        summary[col] = (summary[col] * 100).round(1)
    summary["avg_latency_s"] = summary["avg_latency_s"].round(2)
    unans = df[~df.answerable].groupby("system")["correct"].mean().mul(100).round(1)
    summary["refused_unanswerable_%"] = unans

    table = f"Eval set: {qfile} ({len(questions)} questions), judge: {JUDGE_MODEL}\n\n"
    table += summary.to_markdown()
    Path(f"results/{name}_summary.md").write_text(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
    
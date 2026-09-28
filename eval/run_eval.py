import json
import sys
import time
from pathlib import Path

import pandas as pd

from app.baseline import answer as baseline_answer
from app.graph import ask as agent_answer
from app.llm import chat

JUDGE_MODEL = "gpt-4o"

CORRECTNESS_JUDGE = (
    "You are a strict grader for a Medicare policy Q&A system. You get a QUESTION, whether "
    "it is ANSWERABLE from the documents, a REFERENCE answer, and the system's ANSWER.\n"
    "- If ANSWERABLE is yes: an answer that says 'I don't know' or refuses is NOT correct.\n"
           "- If ANSWERABLE is no: the answer is correct if it declines to answer (e.g. 'I don't "
           "know' or 'the documents do not contain this') and does not invent an answer. "
           "Mentioning related facts that ARE in the documents is fine.\n"
    "Reply in JSON with two booleans: "
    '{"correct": the answer agrees with the REFERENCE on the key facts, '
    '"complete": the answer includes ALL important exceptions or conditions in the REFERENCE}'
)

FAITHFULNESS_JUDGE = (
    "You check whether an ANSWER is supported by the retrieved CONTEXT. Ignore whether the "
    "answer is right in general; only ask: is every factual claim in the ANSWER stated in, "
    "or directly implied by, the CONTEXT? An answer that only says it doesn't know is faithful. "
    'Reply in JSON: {"faithful": true or false}'
)


def judge(q: dict, result: dict) -> dict:
    answerable = q["source"] is not None
    grade = json.loads(chat(
        CORRECTNESS_JUDGE,
        f"QUESTION: {q['question']}\nANSWERABLE: {'yes' if answerable else 'no'}\n"
        f"REFERENCE: {q['reference']}\n\nANSWER:\n{result['answer']}",
        json_mode=True, model=JUDGE_MODEL,
    ))
    faith = json.loads(chat(
        FAITHFULNESS_JUDGE,
        f"CONTEXT:\n{chr(10).join(result['contexts'])[:12000]}\n\nANSWER:\n{result['answer']}",
        json_mode=True, model=JUDGE_MODEL,
    ))
    grade["faithful"] = faith.get("faithful")
    return grade


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

    ans = df[df.answerable]
    summary = ans.groupby("system").agg(
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

    n_ans, n_un = int(ans.shape[0] / 2), int((~df.answerable).sum() / 2)
    table = (f"Eval set: {qfile} ({n_ans} answerable + {n_un} unanswerable), "
             f"judge: {JUDGE_MODEL}\n\n") + summary.to_markdown()
    Path(f"results/{name}_summary.md").write_text(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
    
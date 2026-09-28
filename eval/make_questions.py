"""Drafts eval questions from random chunks. Review the output by hand before using it."""
import json
import random

from app.db import get_conn
from app.llm import chat

N = 40
random.seed(42)

PROMPT = (
    "You write test questions for a Medicare policy Q&A system. From the PASSAGE, write one "
    "question a real beneficiary or claims analyst might ask, which the passage clearly answers. "
    "Avoid questions about section numbers or document structure. Reply in JSON: "
    '{"question": "...", "reference": "the correct answer in one or two sentences, including '
    'any important exception", "usable": true or false (false if the passage is a table of '
    'contents, too fragmentary, or has no clear fact)}'
)


def main():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT source, page, content FROM chunks WHERE length(content) > 800"
        ).fetchall()
    sample = random.sample(rows, N * 2)

    out = []
    for source, page, content in sample:
        q = json.loads(chat(PROMPT, f"PASSAGE ({source}, page {page}):\n{content}", json_mode=True))
        if q.get("usable"):
            out.append({"question": q["question"], "reference": q["reference"],
                        "source": source, "page": page})
            print(f"{len(out)}. {q['question']}")
        if len(out) == N:
            break

    unanswerable = [
        "What is the phone number of the Medicare office in Jersey City?",
        "How much did Medicare spend on advertising in 2025?",
        "Does Medicare cover pet insurance for service animals?",
        "Who was the first administrator of CMS?",
        "What is the Medicare Part B premium in the year 2035?",
    ]
    for q in unanswerable:
        out.append({"question": q, "reference": "Not in the documents - the correct answer is "
                    "that the documents don't contain this.", "source": None, "page": None})

    with open("eval/questions.jsonl", "w") as f:
        for item in out:
            f.write(json.dumps(item) + "\n")
    print(f"\nWrote {len(out)} questions to eval/questions.jsonl")


if __name__ == "__main__":
    main()
    
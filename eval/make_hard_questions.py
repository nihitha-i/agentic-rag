"""Drafts HARD eval questions: plain language + needs two passages from different PDFs.
Review the output by hand before using it."""
import json
import random

from app.db import get_conn
from app.llm import chat

N = 30
random.seed(7)

PROMPT = (
    "You write HARD test questions for a Medicare policy Q&A system. You get two passages "
    "from different documents. Write one question that:\n"
    "1. A real beneficiary or family member would ask in everyday words. Do NOT reuse the "
    "passages' technical terms or section names (say 'nursing home after a hospital stay', "
    "not 'SNF extended care services').\n"
    "2. Needs facts from BOTH passages for a complete answer.\n"
    "Reply in JSON: {\"question\": \"...\", \"reference\": \"the complete correct answer in "
    "2-3 sentences, combining both passages and naming important exceptions\", "
    "\"usable\": true or false (false if the passages don't share a topic, or either is a "
    "table of contents or fragment)}"
)

UNANSWERABLE = [
    "Can my Medicare card be used to pay for my dog's vet bills?",
    "What is the Medicare Part B premium going to be in 2035?",
    "Which hospital in New Jersey has the shortest ER wait time?",
    "Will Medicare pay for my gym membership if I'm under 65 and healthy?",
    "How much did CMS spend on its website redesign?",
]


def find_partner(conn, cid, source):
    """Nearest chunk from a DIFFERENT PDF (None if the index finds none)."""
    return conn.execute(
        """SELECT source, page, content FROM chunks
           WHERE source <> %s AND length(content) > 800
           ORDER BY embedding <=> (SELECT embedding FROM chunks WHERE id = %s)
           LIMIT 1""",
        (source, cid),
    ).fetchone()


def main():
    out = []
    with get_conn() as conn:
        conn.execute("SET hnsw.ef_search = 400")
        ids = [r[0] for r in conn.execute(
            "SELECT id FROM chunks WHERE length(content) > 800").fetchall()]
        random.shuffle(ids)

        for cid in ids:
            a = conn.execute(
                "SELECT source, page, content FROM chunks WHERE id = %s", (cid,)).fetchone()
            b = find_partner(conn, cid, a[0])
            if b is None:
                continue
            q = json.loads(chat(
                PROMPT,
                f"PASSAGE A ({a[0]}, page {a[1]}):\n{a[2]}\n\n"
                f"PASSAGE B ({b[0]}, page {b[1]}):\n{b[2]}",
                json_mode=True, model="gpt-4o"))
            if q.get("usable"):
                out.append({"question": q["question"], "reference": q["reference"],
                            "source": f"{a[0]} p{a[1]} + {b[0]} p{b[1]}", "page": None})
                print(f"{len(out)}. {q['question']}")
            if len(out) == N:
                break

    for q in UNANSWERABLE:
        out.append({"question": q, "reference": "Not in the documents - the correct answer is "
                    "that the documents don't contain this.", "source": None, "page": None})

    with open("eval/questions_hard.jsonl", "w") as f:
        for item in out:
            f.write(json.dumps(item) + "\n")
    print(f"\nWrote {len(out)} questions to eval/questions_hard.jsonl")


if __name__ == "__main__":
    main()
    
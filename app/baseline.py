from app.db import format_context, search
from app.llm import chat

SYSTEM = (
    "Answer the question using only the numbered context. "
    "Cite sources like [1]. If the answer is not in the context, say you don't know."
)


def answer(question: str) -> dict:
    docs = search(question, k=5)
    reply = chat(SYSTEM, f"Context:\n{format_context(docs)}\n\nQuestion: {question}")
    return {"answer": reply, "contexts": [d["content"] for d in docs], "sources": docs}


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Does Medicare cover cosmetic surgery?"
    result = answer(q)
    print(result["answer"])
    print("\nSources:")
    for d in result["sources"]:
        print(f"- {d['source']} page {d['page']} (score {d['score']})")
        
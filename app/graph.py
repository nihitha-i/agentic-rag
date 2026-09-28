import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.db import format_context, search
from app.llm import chat

MAX_ATTEMPTS = 2
PER_QUERY_K = 4
MAX_DOCS = 8


class RAGState(TypedDict, total=False):
    question: str
    route: str
    queries: list
    docs: list
    answer: str
    supported: bool
    feedback: str
    attempts: int


def router(state: RAGState) -> dict:
    """Decide whether the question needs the documents."""
    out = chat(
        "You route questions for a document Q&A system about Medicare. Reply in JSON as "
                '{"route": "general"} ONLY for greetings or small talk. For EVERY other question, '
                'including anything about health care, providers, costs or coverage, reply {"route": "documents"}.',
        state["question"],
        json_mode=True,
    )
    route = json.loads(out).get("route", "documents")
    return {"route": route if route in ("documents", "general") else "documents"}


def general(state: RAGState) -> dict:
    return {
        "answer": "I answer questions about the loaded Medicare documents. Please ask about them.",
        "docs": [],
        "queries": [],
        "supported": True,
    }


def decompose(state: RAGState) -> dict:
    """Split the question into 1-3 focused search queries in Medicare policy terms."""
    feedback = state.get("feedback", "")
    out = chat(
        "You plan searches over Medicare policy documents (Medicare & You handbook and the "
        "Medicare Benefit Policy Manual). Break the user's question into 1-3 short search "
        "queries, one per distinct fact needed. Translate everyday words into official policy "
        "terms (e.g. 'nursing home after a hospital stay' -> 'skilled nursing facility "
        "extended care'). If feedback lists missing facts, add queries for them. "
        'Reply in JSON: {"queries": ["...", "..."]}',
        f"Question: {state['question']}\nFeedback from the last attempt: {feedback or 'none'}",
        json_mode=True,
    )
    queries = [q for q in json.loads(out).get("queries", []) if isinstance(q, str)][:3]
    return {"queries": queries, "attempts": state.get("attempts", 0) + 1}


def retrieve(state: RAGState) -> dict:
    """Search for the original question AND each sub-query, then merge and de-duplicate."""
    seen, merged = set(), []
    for q in [state["question"]] + state["queries"]:
        for d in search(q, k=PER_QUERY_K):
            key = (d["source"], d["page"], d["content"][:80])
            if key not in seen:
                seen.add(key)
                merged.append(d)
    merged.sort(key=lambda d: d["score"], reverse=True)
    return {"docs": merged[:MAX_DOCS]}


def generate(state: RAGState) -> dict:
    reply = chat(
        "Answer using only the numbered context. Answer EVERY part of the question. "
        "Put a citation like [2] after every factual sentence. Include important exceptions "
        "and conditions. If the context does not contain part of the answer, say which part "
        "you could not find.",
        f"Context:\n{format_context(state['docs'])}\n\nQuestion: {state['question']}",
    )
    return {"answer": reply}


def verify(state: RAGState) -> dict:
    """Check every claim is supported AND every part of the question is answered."""
    out = chat(
        "You are a strict fact checker. Check (1) whether every claim in the ANSWER is "
        "supported by the CONTEXT it cites, and (2) whether every part of the QUESTION is "
        "answered. Reply in JSON: "
        '{"supported": true or false, "feedback": "which claims are unsupported or which '
        'facts are missing"}',
        f"CONTEXT:\n{format_context(state['docs'])}\n\nQUESTION: {state['question']}\n\n"
        f"ANSWER:\n{state['answer']}",
        json_mode=True,
    )
    result = json.loads(out)
    supported = bool(result.get("supported", False))
    update = {"supported": supported, "feedback": result.get("feedback", "")}
    if not supported and state["attempts"] >= MAX_ATTEMPTS:
        update["answer"] = (
            "I couldn't find a fully supported answer in the documents. "
            f"Closest attempt:\n\n{state['answer']}"
        )
    return update


def after_verify(state: RAGState) -> str:
    if state["supported"] or state["attempts"] >= MAX_ATTEMPTS:
        return "done"
    return "retry"


def build_graph():
    g = StateGraph(RAGState)
    g.add_node("router", router)
    g.add_node("general", general)
    g.add_node("decompose", decompose)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("verify", verify)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", lambda s: s["route"],
                            {"documents": "decompose", "general": "general"})
    g.add_edge("general", END)
    g.add_edge("decompose", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", after_verify, {"done": END, "retry": "decompose"})
    return g.compile()


GRAPH = build_graph()


def ask(question: str) -> dict:
    s = GRAPH.invoke({"question": question, "attempts": 0})
    docs = s.get("docs", [])
    return {
        "answer": s["answer"],
        "supported": s.get("supported"),
        "attempts": s.get("attempts", 0),
        "search_query": " | ".join(s.get("queries", [])),
        "contexts": [d["content"] for d in docs],
        "sources": docs,
    }


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Does Medicare cover hearing aids?"
    r = ask(q)
    print(r["answer"])
    print(f"\nSub-queries: {r['search_query']}")
    print(f"Verified: {r['supported']} | Attempts: {r['attempts']}")
    print("Sources:")
    for d in r["sources"]:
        print(f"- {d['source']} page {d['page']} (score {d['score']})")
        
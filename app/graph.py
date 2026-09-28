import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.db import format_context, search
from app.llm import chat

MAX_ATTEMPTS = 2


class RAGState(TypedDict, total=False):
    question: str
    route: str
    search_query: str
    docs: list
    answer: str
    supported: bool
    feedback: str
    attempts: int


def router(state: RAGState) -> dict:
    """Decide whether the question needs the documents."""
    out = chat(
        "You route questions for a document Q&A system about Medicare. Reply in JSON as "
        '{"route": "documents"} if answering needs facts from the documents, '
        'or {"route": "general"} for greetings or questions unrelated to documents.',
        state["question"],
        json_mode=True,
    )
    route = json.loads(out).get("route", "documents")
    return {"route": route if route in ("documents", "general") else "documents"}


def general(state: RAGState) -> dict:
    return {
        "answer": "I answer questions about the loaded Medicare documents. Please ask about them.",
        "docs": [],
        "supported": True,
    }


def rewrite(state: RAGState) -> dict:
    """Turn the question (plus any verifier feedback) into a search query."""
    feedback = state.get("feedback", "")
    query = chat(
        "Rewrite the user's question into a short, specific search query for a vector "
        "database of Medicare policy documents. Include official policy terms "
        "(e.g. 'exclusions from coverage'). Return only the query.",
        f"Question: {state['question']}\nFeedback from the last attempt: {feedback or 'none'}",
    )
    return {"search_query": query.strip(), "attempts": state.get("attempts", 0) + 1}


def retrieve(state: RAGState) -> dict:
    k = 5 if state["attempts"] == 1 else 8  # widen the search on a retry
    return {"docs": search(state["search_query"], k=k)}


def generate(state: RAGState) -> dict:
    reply = chat(
        "Answer using only the numbered context. Put a citation like [2] after every "
        "factual sentence. Include important exceptions or conditions. "
        "If the context does not contain the answer, say you don't know.",
        f"Context:\n{format_context(state['docs'])}\n\nQuestion: {state['question']}",
    )
    return {"answer": reply}


def verify(state: RAGState) -> dict:
    """Check every claim in the answer against the cited context."""
    out = chat(
        "You are a strict fact checker. Check whether every claim in the ANSWER is "
        "supported by the CONTEXT it cites, and whether it answers the QUESTION fully. "
        'Reply in JSON: {"supported": true or false, "feedback": "what is missing or unsupported"}',
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
    g.add_node("rewrite", rewrite)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("verify", verify)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", lambda s: s["route"],
                            {"documents": "rewrite", "general": "general"})
    g.add_edge("general", END)
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", after_verify, {"done": END, "retry": "rewrite"})
    return g.compile()


GRAPH = build_graph()


def ask(question: str) -> dict:
    s = GRAPH.invoke({"question": question, "attempts": 0})
    docs = s.get("docs", [])
    return {
        "answer": s["answer"],
        "supported": s.get("supported"),
        "attempts": s.get("attempts", 0),
        "search_query": s.get("search_query"),
        "contexts": [d["content"] for d in docs],
        "sources": docs,
    }


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Does Medicare cover cosmetic surgery?"
    r = ask(q)
    print(r["answer"])
    print(f"\nSearch query used: {r['search_query']}")
    print(f"Verified: {r['supported']} | Attempts: {r['attempts']}")
    print("Sources:")
    for d in r["sources"]:
        print(f"- {d['source']} page {d['page']} (score {d['score']})")
        
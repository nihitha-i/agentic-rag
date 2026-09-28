from fastapi import FastAPI
from pydantic import BaseModel

from app.baseline import answer as baseline_answer
from app.graph import ask

api = FastAPI(title="Agentic RAG - Medicare Policy Assistant")


class Question(BaseModel):
    question: str


@api.get("/health")
def health():
    return {"status": "ok"}


@api.post("/ask")
def ask_agent(q: Question):
    return ask(q.question)


@api.post("/ask/baseline")
def ask_baseline(q: Question):
    return baseline_answer(q.question)

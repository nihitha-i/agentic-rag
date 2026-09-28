import numpy as np
from openai import OpenAI

from app.config import CHAT_MODEL, EMBED_MODEL

client = OpenAI()  # reads OPENAI_API_KEY from the environment


def embed(texts: list[str]) -> list[np.ndarray]:
    """Turn a list of texts into embedding vectors."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [np.array(d.embedding, dtype=np.float32) for d in resp.data]


def chat(system: str, user: str, json_mode: bool = False) -> str:
    """Send one system + user message and return the reply text."""
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **extra,
    )
    return resp.choices[0].message.content

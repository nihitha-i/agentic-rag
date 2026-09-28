import psycopg
from pgvector.psycopg import register_vector

from app.config import DATABASE_URL, EMBED_DIM
from app.llm import embed


def get_conn():
    conn = psycopg.connect(DATABASE_URL, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init_db(conn):
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id BIGSERIAL PRIMARY KEY,
            source TEXT NOT NULL,
            page INT NOT NULL,
            content TEXT NOT NULL,
            embedding vector({EMBED_DIM}) NOT NULL
        )
        """
    )


def search(query: str, k: int = 5) -> list[dict]:
    """Return the k chunks most similar to the query (cosine similarity)."""
    qvec = embed([query])[0]
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT source, page, content, 1 - (embedding <=> %s) AS score
            FROM chunks
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (qvec, qvec, k),
        ).fetchall()
    return [
        {"source": r[0], "page": r[1], "content": r[2], "score": round(float(r[3]), 3)}
        for r in rows
    ]


def format_context(docs: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] ({d['source']}, page {d['page']})\n{d['content']}"
        for i, d in enumerate(docs, start=1)
    )

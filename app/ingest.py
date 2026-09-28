import sys
from pathlib import Path

from pypdf import PdfReader

from app.db import get_conn, init_db
from app.llm import embed

CHUNK_SIZE = 1200  # characters per chunk
OVERLAP = 200      # characters shared between neighbouring chunks
BATCH = 100        # chunks per embedding request


def chunk_text(text: str) -> list[str]:
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - OVERLAP
    return [c for c in chunks if len(c) > 100]


def main(data_dir: str = "data"):
    pdfs = sorted(Path(data_dir).glob("*.pdf"))
    if not pdfs:
        sys.exit("No PDFs found in the data/ folder.")

    conn = get_conn()
    init_db(conn)
    conn.execute("TRUNCATE chunks")

    total = 0
    for pdf in pdfs:
        rows = []
        for page_num, page in enumerate(PdfReader(pdf).pages, start=1):
            for chunk in chunk_text(page.extract_text() or ""):
                rows.append((pdf.name, page_num, chunk))

        for i in range(0, len(rows), BATCH):
            batch = rows[i:i + BATCH]
            vectors = embed([r[2] for r in batch])
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (source, page, content, embedding) VALUES (%s, %s, %s, %s)",
                    [(s, p, c, v) for (s, p, c), v in zip(batch, vectors)],
                )
        total += len(rows)
        print(f"{pdf.name}: {len(rows)} chunks")

    conn.execute(
        "CREATE INDEX IF NOT EXISTS chunks_embedding_idx "
        "ON chunks USING hnsw (embedding vector_cosine_ops)"
    )
    print(f"Done. {total} chunks from {len(pdfs)} PDFs.")


if __name__ == "__main__":
    main()
    
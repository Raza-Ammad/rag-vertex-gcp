import os
from pathlib import Path
from typing import List

import psycopg2
from psycopg2.extensions import connection as PGConnection

import vertexai
from vertexai.language_models import TextEmbeddingModel

# ===== Config =====

# GCP / Vertex
PROJECT_ID = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-004")

# DB (using public IP – same values you already use)
DB_HOST = os.getenv("DB_HOST", "35.184.43.16")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "ragdb")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")  # must be set in env

# Local docs folder
DOCS_DIR = os.getenv("DOCS_DIR", "docs")


# ===== Helpers =====

def get_connection() -> PGConnection:
    if not DB_PASSWORD:
        raise RuntimeError("DB_PASSWORD env var is not set. Export it before running.")
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def init_vertex() -> TextEmbeddingModel:
    if not PROJECT_ID:
        raise RuntimeError("GCP_PROJECT / GOOGLE_CLOUD_PROJECT not set.")
    print("🔧 Initialising Vertex + Cloud SQL...")
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
    return model


def ensure_schema() -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            # pgvector extension + documents table
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    source_uri TEXT NOT NULL,
                    chunk_index INT NOT NULL,
                    content TEXT NOT NULL,
                    embedding VECTOR(768)
                );
                """
            )
        conn.commit()
    print("✅ Ensured documents table exists.")


def chunk_text(text: str, max_chars: int = 800, overlap: int = 200) -> List[str]:
    """
    Simple character-based chunker with overlap.
    Good enough for a first demo.
    """
    text = text.replace("\r\n", "\n")
    chunks: List[str] = []

    start = 0
    length = len(text)

    while start < length:
        end = min(start + max_chars, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap  # move back a bit for overlap
        if start < 0:
            start = 0
        if start >= length:
            break

    return chunks


def get_embedding(model: TextEmbeddingModel, text: str) -> list:
    resp = model.get_embeddings([text])
    return resp[0].values


def ingest_docs(model: TextEmbeddingModel) -> None:
    docs_path = Path(DOCS_DIR)
    if not docs_path.exists():
        print(f"⚠️  Docs directory '{DOCS_DIR}' does not exist. Create it and add .txt files.")
        return

    all_txt_files = list(docs_path.rglob("*.txt"))
    if not all_txt_files:
        print(f"⚠️  No .txt files found under '{DOCS_DIR}'. Add some and re-run.")
        return

    print(f"📂 Found {len(all_txt_files)} .txt files under '{DOCS_DIR}'.")

    with get_connection() as conn:
        with conn.cursor() as cur:
            for file_path in all_txt_files:
                rel_path = str(file_path.relative_to(docs_path))
                print(f"📄 Ingesting {rel_path} ...")

                text = file_path.read_text(encoding="utf-8", errors="ignore")
                chunks = chunk_text(text)

                print(f"   → {len(chunks)} chunks")

                # Clear existing rows for this file (so re-ingest doesn't duplicate)
                cur.execute(
                    "DELETE FROM documents WHERE source_uri = %s;",
                    (rel_path,),
                )

                for idx, chunk in enumerate(chunks):
                    emb = get_embedding(model, chunk)
                    cur.execute(
                        """
                        INSERT INTO documents (source_uri, chunk_index, content, embedding)
                        VALUES (%s, %s, %s, %s);
                        """,
                        (rel_path, idx, chunk, emb),
                    )

        conn.commit()

    print("✅ Finished ingesting all documents into pgvector.")


def main() -> None:
    try:
        model = init_vertex()
        ensure_schema()
        ingest_docs(model)
        print("🎉 Done. Your real documents are now in Cloud SQL pgvector.")
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")


if __name__ == "__main__":
    main()
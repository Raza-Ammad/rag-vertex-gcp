import os
import psycopg2
from typing import List

import vertexai
from vertexai.language_models import TextEmbeddingModel

# ---------------------------------------------------------------------
# Config from environment (works on both Mac + Cloud Shell)
# ---------------------------------------------------------------------
GCP_PROJECT = os.environ.get("GCP_PROJECT", "starry-journal-480011-m8")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")

DB_HOST = os.environ.get("DB_HOST", "35.184.43.16")
DB_PORT = int(os.environ.get("DB_PORT", "5432"))
DB_NAME = os.environ.get("DB_NAME", "ragdb")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")

# Updated embedding model
EMBEDDING_MODEL_NAME = "text-embedding-004"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def init_vertex() -> TextEmbeddingModel:
    """Initialise Vertex AI and load the embedding model."""
    vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
    model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
    return model


def get_connection():
    """Plain psycopg2 connection to Cloud SQL via public IP."""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        connect_timeout=10,
    )


def to_pgvector_literal(vec: List[float]) -> str:
    """
    Convert a Python list of floats into a pgvector literal string: [1.0,2.0,...]
    """
    return "[" + ",".join(str(x) for x in vec) + "]"


def get_embedding(model: TextEmbeddingModel, text: str) -> List[float]:
    """Call Vertex AI embedding model and return a float vector."""
    resp = model.get_embeddings([text])
    # Vertex returns a list of one embedding for a single input
    return resp[0].values


# ---------------------------------------------------------------------
# DB schema + sample row
# ---------------------------------------------------------------------
def init_schema():
    """Ensure the pgvector extension + documents table exist."""
    print("✅ Ensuring documents table exists...")
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Enable pgvector
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # text-embedding-004 returns 768-dim vectors
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    source_uri TEXT,
                    chunk_index INT,
                    content TEXT,
                    embedding vector(768)
                );
                """
            )
        conn.commit()


def insert_sample_row(embed_model: TextEmbeddingModel):
    """Insert a single sample row so we can test RAG."""
    sample_text = (
        "This is a small sample document explaining what a RAG system does. "
        "It stores text chunks with vector embeddings in pgvector and uses "
        "Vertex Gemini to answer questions based on those chunks."
    )

    embedding = get_embedding(embed_model, sample_text)
    pgvec = to_pgvector_literal(embedding)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (source_uri, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s::vector);
                """,
                ("sample_doc.txt", 0, sample_text, pgvec),
            )
        conn.commit()

    print("✅ Inserted 1 sample row into documents table.")
    print("🎉 Done. You now have one embedded row in Cloud SQL pgvector.")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
def main():
    print("🔧 Initialising Vertex + Cloud SQL...")

    # Init embedding model
    embed_model = init_vertex()

    # Ensure DB schema + sample data
    init_schema()
    insert_sample_row(embed_model)


if __name__ == "__main__":
    main()
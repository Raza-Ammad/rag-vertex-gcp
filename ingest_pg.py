import os
import psycopg2
from psycopg2.extras import Json
from typing import List
import vertexai
from vertexai.language_models import TextEmbeddingModel

# --- Config from environment ---
GCP_PROJECT = os.environ["GCP_PROJECT"]
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")

DB_HOST = os.environ["DB_HOST"]
DB_PORT = int(os.environ.get("DB_PORT", "5432"))
DB_NAME = os.environ["DB_NAME"]
DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]

EMBEDDING_MODEL_NAME = "text-embedding-004"


def init_vertex() -> TextEmbeddingModel:
    """Initialise Vertex AI and return the embedding model."""
    vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
    print("🔧 Initialising Vertex + Cloud SQL...")
    model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
    return model


def get_connection():
    """Plain psycopg2 connection to Cloud SQL using PUBLIC IP."""
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        connect_timeout=10,
        sslmode="require",  # optional but recommended
    )
    return conn


def get_embedding(model: TextEmbeddingModel, text: str) -> List[float]:
    """Call Vertex embeddings API and return the first embedding vector."""
    emb = model.get_embeddings([text])[0]
    return emb.values  # list[float]


def init_schema():
    """Create pgvector extension + documents table if not already present."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Enable pgvector
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # 768 dims is correct for text-embedding-004
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
    print("✅ Ensured documents table exists.")


def insert_sample_row(model: TextEmbeddingModel):
    """Insert a single sample row with an embedding into documents."""
    text = "This is a small sample document describing a retrieval-augmented generation demo."
    emb = get_embedding(model, text)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (source_uri, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s);
                """,
                (
                    "sample://demo",
                    0,
                    text,
                    emb,  # psycopg2 + pgvector understand list[float]
                ),
            )
        conn.commit()
    print("✅ Inserted 1 sample row into documents table.")
    print("🎉 Done. You now have one embedded row in Cloud SQL pgvector.")


def main():
    model = init_vertex()
    init_schema()
    insert_sample_row(model)


if __name__ == "__main__":
    main()
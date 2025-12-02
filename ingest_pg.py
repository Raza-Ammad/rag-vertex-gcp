import os
from dotenv import load_dotenv

import psycopg2
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

# Load environment variables from .env
load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT", "starry-journal-480011-m8")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "ragdb")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# 768 dims for text embedding model
EMBEDDING_DIM = 768


def get_connection():
    """Connect to Cloud SQL Postgres with SSL over public IP."""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        sslmode="require",
    )


def init_vertex():
    """Initialise Vertex AI and return a text embedding model."""
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    # textembedding-gecko@001 -> 768-dim embeddings suitable for RAG
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
    return model


def get_embedding(model: TextEmbeddingModel, text: str):
    """Return a list[float] embedding for given text."""
    inp = TextEmbeddingInput(text=text, task_type="RETRIEVAL_DOCUMENT")
    embeddings = model.get_embeddings([inp])
    return embeddings[0].values  # list of floats


def init_schema():
    """Create documents table with a pgvector column if it doesn't exist."""
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        source_uri TEXT,
        chunk_index INT,
        content TEXT,
        embedding vector({EMBEDDING_DIM})
    );
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(create_table_sql)
            conn.commit()
    print("✅ Ensured documents table exists.")


def insert_sample_row(model: TextEmbeddingModel):
    text = (
        "This is a sample document chunk for testing Vertex AI embeddings "
        "with pgvector in Cloud SQL."
    )
    embedding = get_embedding(model, text)

    # Convert embedding to pgvector string literal: '[0.1,0.2,...]'
    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

    insert_sql = """
    INSERT INTO documents (source_uri, chunk_index, content, embedding)
    VALUES (%s, %s, %s, %s::vector);
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                insert_sql,
                ("sample", 0, text, embedding_str),
            )
            conn.commit()
    print("✅ Inserted 1 sample row into documents table.")


def main():
    print("🔧 Initialising Vertex + Cloud SQL...")
    model = init_vertex()
    init_schema()
    insert_sample_row(model)
    print("🎉 Done. You now have one embedded row in Cloud SQL pgvector.")


if __name__ == "__main__":
    main()
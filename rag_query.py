import os
from typing import List, Tuple

from dotenv import load_dotenv

import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel

from google.cloud.sql.connector import Connector, IPTypes
import pg8000


# Load .env if present
load_dotenv()

# ---- Config ----
PROJECT_ID = os.getenv("GCP_PROJECT", "starry-journal-480011-m8")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")

INSTANCE_CONNECTION_NAME = os.environ.get("INSTANCE_CONNECTION_NAME")
DB_NAME = os.environ.get("DB_NAME", "ragdb")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD")

EMBEDDING_MODEL_NAME = "text-embedding-004"
EMBEDDING_DIM = 768
GEMINI_MODEL_NAME = "gemini-1.5-flash-001"


# ---- DB helpers ----
def get_connection():
    """
    Connect to Cloud SQL Postgres using Cloud SQL Python Connector (pg8000).
    Uses INSTANCE_CONNECTION_NAME + DB_* env vars.
    """
    if not INSTANCE_CONNECTION_NAME:
        raise RuntimeError("INSTANCE_CONNECTION_NAME env var is not set")

    connector = Connector()

    conn: pg8000.dbapi.Connection = connector.connect(
        INSTANCE_CONNECTION_NAME,
        "pg8000",
        user=DB_USER,
        password=DB_PASSWORD,
        db=DB_NAME,
        ip_type=IPTypes.PUBLIC,
    )
    return conn


# ---- Vertex helpers ----
def init_vertex():
    """Initialise Vertex AI and return (embedding_model, qa_model)."""
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    embed_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
    qa_model = GenerativeModel(GEMINI_MODEL_NAME)
    return embed_model, qa_model


def get_embedding(embed_model: TextEmbeddingModel, text: str):
    """Return a list[float] embedding for given text using text-embedding-004."""
    embeddings = embed_model.get_embeddings([text])
    return embeddings[0].values


# ---- RAG core ----
def search_similar_chunks(
    embed_model: TextEmbeddingModel,
    question: str,
    top_k: int = 5,
) -> List[Tuple[int, str, int, str, float]]:
    """
    Embed the question, query pgvector, and return top_k rows as:
    (id, source_uri, chunk_index, content, distance)
    """
    query_embedding = get_embedding(embed_model, question)
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    sql = """
    SELECT
        id,
        source_uri,
        chunk_index,
        content,
        embedding <-> %s::vector AS distance
    FROM documents
    ORDER BY embedding <-> %s::vector
    LIMIT %s;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (embedding_str, embedding_str, top_k))
            rows = cur.fetchall()

    return rows


def build_context(chunks: List[Tuple[int, str, int, str, float]]) -> str:
    """Format retrieved chunks into a context string for Gemini."""
    if not chunks:
        return "No relevant context found in the database."

    parts = []
    for row in chunks:
        _id, source_uri, chunk_index, content, distance = row
        parts.append(
            f"[source={source_uri}, chunk={chunk_index}, distance={distance:.4f}]\n{content}"
        )
    return "\n\n---\n\n".join(parts)


def answer_question(
    qa_model: GenerativeModel,
    question: str,
    context: str,
) -> str:
    """Call Gemini with the retrieved context + user question."""
    prompt = f"""
You are a helpful assistant answering questions about documents stored in a pgvector database.

You must:
- Use ONLY the information in the context.
- If the context is not sufficient, say you don't have enough information.
- Be concise and clear.

Context:
{context}

Question:
{question}
"""
    response = qa_model.generate_content(prompt)
    return response.text


def main():
    print("🔧 Initialising Vertex AI (embeddings + Gemini)...")
    embed_model, qa_model = init_vertex()

    print("❓ Enter your question (or just press Enter to quit):")
    question = input("> ").strip()
    if not question:
        print("No question entered. Exiting.")
        return

    print("🔍 Searching for relevant chunks in pgvector...")
    chunks = search_similar_chunks(embed_model, question, top_k=5)

    if not chunks:
        print("No rows found in the documents table yet.")
        return

    context = build_context(chunks)

    print("\n📚 Retrieved context:\n")
    print(context)
    print("\n🤖 Gemini answer:\n")

    answer = answer_question(qa_model, question, context)
    print(answer)


if __name__ == "__main__":
    main()
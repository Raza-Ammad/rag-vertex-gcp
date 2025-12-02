import os
from typing import List, Dict, Any

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai import generative_models

load_dotenv()

# ======== CONFIG ========
GCP_PROJECT = os.environ.get("GCP_PROJECT", "starry-journal-480011-m8")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")

# Models you’re using now
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-004")
GENERATION_MODEL_NAME = os.environ.get("GENERATION_MODEL_NAME", "gemini-2.0-flash")

DB_HOST = os.environ.get("DB_HOST", "35.184.43.16")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "ragdb")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")


# ======== DB CONNECTION ========
def get_connection() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


# ======== VERTEX INIT ========
def init_vertex():
    """Initialise Vertex AI and return (embedding_model, qa_model)."""
    vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)

    embed_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
    qa_model = generative_models.GenerativeModel(GENERATION_MODEL_NAME)

    return embed_model, qa_model


def get_embedding(model: TextEmbeddingModel, text: str) -> List[float]:
    """Return a single embedding vector for the given text."""
    resp = model.get_embeddings([text])
    return resp[0].values


# ======== RAG PIPELINE ========
def search_similar_chunks(embed_model: TextEmbeddingModel, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Return top_k most similar chunks from pgvector."""
    query_emb = get_embedding(embed_model, query)

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    id,
                    source_uri,
                    chunk_index,
                    content,
                    embedding <-> %s::vector AS distance
                FROM documents
                ORDER BY embedding <-> %s::vector
                LIMIT %s;
                """,
                # key fix: CAST the parameter to vector in SQL using ::vector
                (query_emb, query_emb, top_k),
            )
            rows = cur.fetchall()

    return rows


def build_context(chunks: List[Dict[str, Any]]) -> str:
    """Build a text context block from retrieved chunks."""
    parts = []
    for row in chunks:
        parts.append(
            f"[source={row['source_uri']} chunk={row['chunk_index']} "
            f"distance={row['distance']:.4f}]\n"
            f"{row['content']}\n"
        )
    return "\n\n---\n\n".join(parts)


def answer_question(qa_model: generative_models.GenerativeModel, question: str, context: str) -> str:
    """Use Gemini to answer a question given retrieved context."""
    prompt = f"""
You are a helpful assistant doing retrieval-augmented QA.

Use only the context below to answer the user's question.
If the answer isn't in the context, say that you don't know.

CONTEXT:
{context}

QUESTION:
{question}
"""
    resp = qa_model.generate_content(prompt)
    return resp.text


# ======== CLI LOOP ========
def main():
    print("🔧 Initialising Vertex AI (embeddings + Gemini)...")
    embed_model, qa_model = init_vertex()

    while True:
        print("❓ Enter your question (or just press Enter to quit):")
        question = input("> ").strip()
        if not question:
            print("👋 Bye!")
            break

        try:
            print("🔍 Searching for relevant chunks in pgvector...")
            chunks = search_similar_chunks(embed_model, question, top_k=5)

            if not chunks:
                print("⚠️ No chunks found in the database yet. Did you run ingest_pg.py?")
                continue

            print("\nTop retrieved chunks:")
            for i, row in enumerate(chunks, start=1):
                snippet = row["content"][:120].replace("\n", " ")
                print(
                    f"{i}. source={row['source_uri']} "
                    f"chunk={row['chunk_index']} "
                    f"distance={row['distance']:.4f}\n"
                    f"   {snippet}..."
                )

            context = build_context(chunks)

            print("\n🤖 Generating answer with Gemini...")
            answer = answer_question(qa_model, question, context)

            print("\n===== ANSWER =====")
            print(answer)
            print("==================\n")

        except Exception as e:
            print(f"❌ Error while running query: {e}")


if __name__ == "__main__":
    main()
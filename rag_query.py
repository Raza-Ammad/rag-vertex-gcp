import os
import textwrap
from typing import List

import psycopg2
from psycopg2.extras import RealDictCursor

import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel

# ========= Config =========

PROJECT_ID = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-004")
QA_MODEL_NAME = os.getenv("QA_MODEL_NAME", "gemini-2.0-flash")

DB_HOST = os.getenv("DB_HOST", "35.184.43.16")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "ragdb")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")


# ========= Helpers =========

def get_connection() -> psycopg2.extensions.connection:
    if not DB_PASSWORD:
        raise RuntimeError("DB_PASSWORD env var is not set.")
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def init_vertex():
    if not PROJECT_ID:
        raise RuntimeError("GCP_PROJECT / GOOGLE_CLOUD_PROJECT not set.")

    print("🔧 Initialising Vertex AI (embeddings + Gemini)...")
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    embed_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
    qa_model = GenerativeModel(QA_MODEL_NAME)
    return embed_model, qa_model


def get_embedding(model: TextEmbeddingModel, text: str) -> List[float]:
    resp = model.get_embeddings([text])
    return resp[0].values


def search_similar_chunks(embed_model: TextEmbeddingModel, query: str, top_k: int = 5):
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
                    embedding <-> %s AS distance
                FROM documents
                ORDER BY embedding <-> %s
                LIMIT %s;
                """,
                (query_emb, query_emb, top_k),
            )
            rows = cur.fetchall()

    return rows


def build_context(chunks) -> str:
    """Build a text context block from the retrieved chunks."""
    parts = []
    for ch in chunks:
        parts.append(
            f"Source: {ch['source_uri']} (chunk {ch['chunk_index']})\n{ch['content']}\n"
        )
    return "\n\n---\n\n".join(parts)


def answer_question(qa_model: GenerativeModel, question: str, context: str) -> str:
    """Ask Gemini to answer based ONLY on the retrieved context."""
    prompt = textwrap.dedent(
        f"""
        You are a helpful assistant for a Retrieval Augmented Generation (RAG) system.

        Answer the user's question **only** using the information in the context below.
        If the context is not sufficient or does not contain the answer, reply:
        "I don't know based on the stored documents."

        Do not invent new facts. Be concise and clear.

        Context:
        {context}

        Question: {question}

        Answer in 3–6 sentences.
        """
    )

    resp = qa_model.generate_content(prompt)
    return resp.text.strip()


# ========= Main CLI =========

def main():
    try:
        embed_model, qa_model = init_vertex()
    except Exception as e:
        print(f"❌ Failed to init Vertex AI: {e}")
        return

    while True:
        print("❓ Enter your question (or just press Enter to quit):")
        question = input("> ").strip()
        if not question:
            print("👋 Bye.")
            break

        try:
            print("🔍 Searching for relevant chunks in pgvector...")
            chunks = search_similar_chunks(embed_model, question, top_k=5)

            if not chunks:
                print("⚠️ No relevant chunks found in the database.")
                continue

            print("📎 Top chunks:")
            for i, ch in enumerate(chunks, start=1):
                snippet = ch["content"][:120].replace("\n", " ")
                print(
                    f"  {i}. [{ch['source_uri']} #{ch['chunk_index']}] "
                    f"{snippet}..."
                )

            context = build_context(chunks)
            print("🤖 Asking Gemini with retrieved context...")
            answer = answer_question(qa_model, question, context)

            print("\n====== ANSWER ======")
            print(answer)
            print("====================\n")

        except Exception as e:
            print(f"❌ Error while running query: {e}")


if __name__ == "__main__":
    main()
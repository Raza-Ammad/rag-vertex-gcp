import os
import psycopg2
from typing import List, Dict, Any

import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel

# ---------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------
GCP_PROJECT = os.environ.get("GCP_PROJECT", "starry-journal-480011-m8")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")

DB_HOST = os.environ.get("DB_HOST", "35.184.43.16")
DB_PORT = int(os.environ.get("DB_PORT", "5432"))
DB_NAME = os.environ.get("DB_NAME", "ragdb")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")

# Models
EMBEDDING_MODEL_NAME = "text-embedding-004"
QA_MODEL_NAME = "gemini-1.5-flash"  # this was working in your earlier tests


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def init_vertex():
    """Initialise Vertex AI + load embedding and QA models."""
    vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)

    embed_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
    qa_model = GenerativeModel(QA_MODEL_NAME)

    return embed_model, qa_model


def get_connection():
    """Plain psycopg2 connection."""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        connect_timeout=10,
    )


def to_pgvector_literal(vec: List[float]) -> str:
    """Convert Python list to pgvector literal: [1.0,2.0,...]."""
    return "[" + ",".join(str(x) for x in vec) + "]"


def get_embedding(model: TextEmbeddingModel, text: str) -> List[float]:
    """Call Vertex AI embedding model and get vector."""
    resp = model.get_embeddings([text])
    return resp[0].values


def search_similar_chunks(embed_model: TextEmbeddingModel, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Embed the query, search pgvector for nearest chunks."""
    query_emb = get_embedding(embed_model, query)
    pgvec = to_pgvector_literal(query_emb)

    with get_connection() as conn:
        with conn.cursor() as cur:
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
                (pgvec, pgvec, top_k),
            )
            rows = cur.fetchall()

    results = []
    for row in rows:
        results.append(
            {
                "id": row[0],
                "source_uri": row[1],
                "chunk_index": row[2],
                "content": row[3],
                "distance": float(row[4]),
            }
        )
    return results


def build_context(chunks: List[Dict[str, Any]]) -> str:
    """Turn chunks into a text block for the LLM."""
    parts = []
    for ch in chunks:
        parts.append(
            f"[id={ch['id']} src={ch['source_uri']} idx={ch['chunk_index']} "
            f"dist={ch['distance']:.4f}]\n{ch['content']}"
        )
    return "\n\n".join(parts)


def answer_question(qa_model: GenerativeModel, question: str, context: str) -> str:
    """Ask Gemini to answer based only on the retrieved context."""
    prompt = f"""
You are a RAG question-answering assistant.

You are given some context chunks retrieved from a pgvector database and a user question.
Answer the question **only** using the information in the context.
If the context is not sufficient, say that you are not sure.

CONTEXT:
{context}

QUESTION:
{question}

Please give a clear and concise answer, ideally 2–4 sentences.
"""
    resp = qa_model.generate_content(prompt)
    return resp.text


# ---------------------------------------------------------------------
# Entry point (CLI loop)
# ---------------------------------------------------------------------
def main():
    print("🔧 Initialising Vertex AI (embeddings + Gemini)...")
    embed_model, qa_model = init_vertex()

    while True:
        print("❓ Enter your question (or just press Enter to quit):")
        question = input("> ").strip()
        if not question:
            print("👋 Bye!")
            break

        print("🔍 Searching for relevant chunks in pgvector...")
        chunks = search_similar_chunks(embed_model, question, top_k=5)

        if not chunks:
            print("No matching documents found in the database.")
            continue

        context = build_context(chunks)
        answer = answer_question(qa_model, question, context)

        print("\n📎 Retrieved chunks:")
        for ch in chunks:
            print(
                f"- id={ch['id']} | src={ch['source_uri']} "
                f"| idx={ch['chunk_index']} | dist={ch['distance']:.4f}"
            )

        print("\n🧠 Answer:")
        print(answer)
        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()
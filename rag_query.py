import os
import psycopg2

import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
PROJECT_ID = os.getenv("GCP_PROJECT", "starry-journal-480011-m8")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")

# Vertex AI models
EMBEDDING_MODEL_NAME = "textembedding-gecko@001"
GENERATION_MODEL_NAME = "gemini-1.5-flash-001"

# Direct Cloud SQL connection (PUBLIC IP)
DB_HOST = os.getenv("DB_HOST", "35.184.43.16")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "ragdb")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")


# -----------------------------------------------------------------------------
# Vertex AI helpers
# -----------------------------------------------------------------------------
def init_vertex():
    """Initialise Vertex AI embeddings + Gemini models."""
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    embed_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
    qa_model = GenerativeModel(GENERATION_MODEL_NAME)
    return embed_model, qa_model


def get_embedding(model: TextEmbeddingModel, text: str):
    """Return a single embedding vector for a text string."""
    resp = model.get_embeddings([text])
    return resp[0].values


# -----------------------------------------------------------------------------
# Postgres / pgvector helpers
# -----------------------------------------------------------------------------
def get_connection():
    """Plain psycopg2 connection to Cloud SQL via public IP."""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        sslmode="require",  # good default for Cloud SQL
    )


def search_similar_chunks(embed_model, query: str, top_k: int = 5):
    """Do a pgvector similarity search against the documents table."""
    query_emb = get_embedding(embed_model, query)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, source_uri, chunk_index, content
                FROM documents
                ORDER BY embedding <-> %s
                LIMIT %s;
                """,
                (query_emb, top_k),
            )
            rows = cur.fetchall()

    chunks = []
    for row in rows:
        chunks.append(
            {
                "id": row[0],
                "source_uri": row[1],
                "chunk_index": row[2],
                "content": row[3],
            }
        )
    return chunks


# -----------------------------------------------------------------------------
# RAG prompt + answer
# -----------------------------------------------------------------------------
def build_context(chunks):
    """Turn retrieved rows into a context block for Gemini."""
    parts = []
    for i, ch in enumerate(chunks, start=1):
        snippet = ch["content"]
        parts.append(
            f"Chunk {i} (source={ch['source_uri']}, idx={ch['chunk_index']}):\n"
            f"{snippet}\n"
        )
    return "\n---\n".join(parts)


def answer_question(qa_model, question: str, context: str) -> str:
    """Call Gemini with the retrieved context + question."""
    prompt = f"""
You are a helpful assistant answering questions using the provided context only.

Context:
{context}

Question: {question}

Answer clearly. If the context does not contain the answer, say you don't
know based on the documents.
"""
    resp = qa_model.generate_content(prompt)
    return resp.candidates[0].content.parts[0].text


# -----------------------------------------------------------------------------
# CLI loop
# -----------------------------------------------------------------------------
def main():
    print("🔧 Initialising Vertex AI (embeddings + Gemini)...")
    embed_model, qa_model = init_vertex()

    while True:
        print("❓ Enter your question (or just press Enter to quit):")
        question = input("> ").strip()
        if not question:
            break

        print("🔍 Searching for relevant chunks in pgvector...")
        chunks = search_similar_chunks(embed_model, question, top_k=5)

        if not chunks:
            print("No matching chunks found in documents table.")
            continue

        context = build_context(chunks)

        print("\nRetrieved context:\n")
        print(context)

        print("\n🤖 Gemini answer:\n")
        answer = answer_question(qa_model, question, context)
        print(answer)
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
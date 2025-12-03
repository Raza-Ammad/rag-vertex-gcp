import os
import time
from typing import List, Dict, Any

import streamlit as st
import psycopg2
from psycopg2 import pool, extras
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
import vertexai
from google.cloud import storage  # <--- NEW IMPORT

from PyPDF2 import PdfReader

# ---------- CONFIG / ENV ----------

GCP_PROJECT = os.environ.get("GCP_PROJECT", "starry-journal-480011-m8")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
# Set your bucket name here or in your environment variables
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "rag-docs-raza") 

DB_HOST = os.environ.get("DB_HOST", "35.184.43.16")
DB_PORT = int(os.environ.get("DB_PORT", "5432"))
DB_NAME = os.environ.get("DB_NAME", "ragdb")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "Bhatti@512")

EMBEDDING_MODEL_NAME = "text-embedding-004"
GENERATION_MODEL_NAME = "gemini-2.0-flash"

EMBEDDING_BATCH_SIZE = 10 

# ---------- DB HELPERS ----------

@st.cache_resource
def get_db_pool():
    return psycopg2.pool.SimpleConnectionPool(
        1, 20,
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        connect_timeout=10
    )

def execute_query(query: str, params: tuple = None, fetch: bool = False, execute_values_data: List = None):
    db_pool = get_db_pool()
    conn = db_pool.getconn()
    res = None
    try:
        with conn.cursor() as cur:
            if execute_values_data:
                extras.execute_values(cur, query, execute_values_data)
            else:
                cur.execute(query, params)
            if fetch:
                res = cur.fetchall()
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        db_pool.putconn(conn)
    return res

def ensure_db_schema():
    execute_query("CREATE EXTENSION IF NOT EXISTS vector;")
    execute_query("""
        CREATE TABLE IF NOT EXISTS documents (
            id          BIGSERIAL PRIMARY KEY,
            source_uri  TEXT,
            chunk_index INT,
            content     TEXT,
            embedding   vector(768)
        );
    """)
    execute_query("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_indexes 
                WHERE tablename = 'documents' AND indexname = 'docs_embedding_hnsw'
            ) THEN
                CREATE INDEX docs_embedding_hnsw ON documents 
                USING hnsw (embedding vector_cosine_ops);
            END IF;
        END $$;
    """)

def vector_to_pgarray(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

def list_documents() -> List[Dict]:
    rows = execute_query("""
        SELECT source_uri, COUNT(*) 
        FROM documents 
        GROUP BY source_uri 
        ORDER BY source_uri;
    """, fetch=True)
    return [{"source_uri": r[0], "chunk_count": r[1]} for r in rows]

# ---------- VERTEX AI & GCS HELPERS ----------

@st.cache_resource(show_spinner=False)
def init_models():
    if not GCP_PROJECT:
        raise RuntimeError("GCP_PROJECT environment variable is not set.")
    vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
    embed_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
    qa_model = GenerativeModel(GENERATION_MODEL_NAME)
    return embed_model, qa_model

def get_embeddings_batched(embed_model: TextEmbeddingModel, texts: List[str]) -> List[List[float]]:
    all_embeddings = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        try:
            embeddings = embed_model.get_embeddings(batch)
            all_embeddings.extend([list(e.values) for e in embeddings])
        except Exception as e:
            st.error(f"Error embedding batch {i}: {e}")
            raise e
    return all_embeddings

def upload_file_to_bucket(file_obj, filename: str) -> str:
    """Uploads the file object to GCS and returns the gs:// URI."""
    client = storage.Client(project=GCP_PROJECT)
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(filename)
    
    # Reset pointer to start just in case
    file_obj.seek(0)
    blob.upload_from_file(file_obj)
    
    # Reset pointer again so the PDF reader can read it later
    file_obj.seek(0)
    
    return f"gs://{GCS_BUCKET_NAME}/{filename}"

# ---------- TEXT PROCESSING ----------

def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    parts = []
    for page in reader.pages:
        txt = page.extract_text()
        if txt:
            parts.append(txt)
    return "\n".join(parts)

def chunk_text(text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + max_chars
        if end < text_len:
            last_space = text.rfind(' ', start, end)
            if last_space != -1 and last_space > start:
                end = last_space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start >= end:
            start = end
    return chunks

def ingest_document(embed_model: TextEmbeddingModel, source_uri: str, text: str):
    """Chunks text, embeds in BATCHES, and inserts via BULK insert."""
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No text content found.")

    st.info(f"Generated {len(chunks)} chunks. Generating embeddings...")
    embeddings = get_embeddings_batched(embed_model, chunks)

    # Delete existing entries for this URI to avoid duplicates
    execute_query("DELETE FROM documents WHERE source_uri = %s", (source_uri,))

    insert_data = []
    for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        insert_data.append((source_uri, idx, chunk, vector_to_pgarray(emb)))

    query = """
        INSERT INTO documents (source_uri, chunk_index, content, embedding)
        VALUES %s
    """
    execute_query(query, execute_values_data=insert_data)

# ---------- RETRIEVAL + GENERATION ----------

def search_similar_chunks(embed_model, query, top_k=5, source_filter=None):
    query_emb = get_embeddings_batched(embed_model, [query])[0]
    
    base_query = """
        SELECT id, source_uri, chunk_index, content
        FROM documents
    """
    params = [vector_to_pgarray(query_emb), top_k]
    
    if source_filter and source_filter != "__ALL__":
        where_clause = "WHERE source_uri = %s"
        params.insert(0, source_filter)
        full_sql = f"{base_query} {where_clause} ORDER BY embedding <-> %s::vector LIMIT %s"
    else:
        full_sql = f"{base_query} ORDER BY embedding <-> %s::vector LIMIT %s"

    rows = execute_query(full_sql, params=tuple(params), fetch=True)
    return [
        {"id": r[0], "source_uri": r[1], "chunk_index": r[2], "content": r[3]}
        for r in rows
    ]

def answer_question(qa_model: GenerativeModel, question: str, chunks: List[Dict]) -> str:
    if not chunks:
        return "I couldn't find any relevant information in the documents."

    context_str = "\n\n".join([
        f"Source: {c['source_uri']} (Chunk {c['chunk_index']})\nContent: {c['content']}" 
        for c in chunks
    ])

    prompt = f"""
    You are an intelligent assistant. Answer the user's question using ONLY the provided context.
    
    CONTEXT:
    {context_str}
    
    USER QUESTION:
    {question}
    
    If the answer is not in the context, state that you do not know.
    """
    response = qa_model.generate_content(prompt)
    return response.text

# ---------- STREAMLIT UI ----------

def main():
    st.set_page_config("Vertex RAG", layout="wide")
    st.title("🚀 RAG (Vertex AI + GCS Bucket + Cloud SQL)")

    try:
        ensure_db_schema()
        embed_model, qa_model = init_models()
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        st.stop()

    with st.sidebar:
        st.header("Upload Document")
        st.caption(f"Storage: gs://{GCS_BUCKET_NAME}")
        uploaded_file = st.file_uploader("Drop PDF/TXT", type=["txt", "pdf"])
        
        if uploaded_file and st.button("Ingest Document"):
            try:
                with st.spinner("Uploading to GCS & Processing..."):
                    # 1. Upload to GCS Bucket
                    gcs_uri = upload_file_to_bucket(uploaded_file, uploaded_file.name)
                    st.toast(f"Uploaded to {gcs_uri}")

                    # 2. Extract Text
                    if uploaded_file.name.endswith(".pdf"):
                        text = extract_text_from_pdf(uploaded_file)
                    else:
                        text = uploaded_file.read().decode("utf-8", errors="ignore")
                    
                    # 3. Ingest (Embed & Store in DB)
                    ingest_document(embed_model, gcs_uri, text)
                    
                st.success("Ingestion Complete!")
                time.sleep(1) 
                st.rerun() # Refresh to show new doc in list
            except Exception as e:
                st.error(f"Error: {e}")

        st.markdown("---")
        st.header("Database Stats")
        docs = list_documents()
        doc_options = ["__ALL__"] + [d["source_uri"] for d in docs]
        
        if docs:
            for d in docs:
                st.caption(f"📄 {d['source_uri']}")
        
        filter_doc = st.selectbox("Filter Search", doc_options)
        top_k = st.slider("Retrieval Count", 1, 20, 5)

    question = st.text_input("Ask a question about your documents:")
    
    if question:
        with st.spinner("Retrieving and Generating..."):
            chunks = search_similar_chunks(embed_model, question, top_k, filter_doc)
            answer = answer_question(qa_model, question, chunks)
            
            st.markdown("### Answer")
            st.write(answer)
            
            with st.expander("View Retrieved Context"):
                for c in chunks:
                    st.info(f"**{c['source_uri']}**: {c['content']}")

if __name__ == "__main__":
    main()
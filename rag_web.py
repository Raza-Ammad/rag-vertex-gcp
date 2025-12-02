import os
import textwrap
from typing import List, Tuple

import psycopg2
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from psycopg2.extras import register_default_json, register_default_jsonb

import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel

# ---------- Config from environment ----------

PROJECT_ID = os.environ.get("GCP_PROJECT", "starry-journal-480011-m8")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")

DB_HOST = os.environ.get("DB_HOST", "35.184.43.16")
DB_PORT = int(os.environ.get("DB_PORT", "5432"))
DB_NAME = os.environ.get("DB_NAME", "ragdb")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "your_password_here")  # override via env

EMBEDDING_MODEL_NAME = "text-embedding-004"
GENERATION_MODEL_NAME = "gemini-2.0-flash"

# Register JSON handlers (avoids some psycopg warnings)
register_default_json(loads=lambda x: x)
register_default_jsonb(loads=lambda x: x, globally=True)

# ---------- Vertex + DB helpers ----------

def init_vertex():
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    embed_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
    qa_model = GenerativeModel(GENERATION_MODEL_NAME)
    return embed_model, qa_model


def get_db_conn():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def ensure_schema():
    """Make sure the documents table exists."""
    create_sql = """
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        source_uri TEXT,
        chunk_index INT,
        content TEXT,
        embedding vector(768)
    );
    """
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(create_sql)
        conn.commit()


def chunk_text(text: str, max_chars: int = 800) -> List[str]:
    """Naive chunking by characters, splitting on paragraph boundaries where possible."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        # If single paragraph itself is huge, hard-wrap it
        if len(para) > max_chars:
            wrapped = textwrap.wrap(para, width=max_chars)
        else:
            wrapped = [para]

        for piece in wrapped:
            if len(current) + len(piece) + 1 <= max_chars:
                current = (current + "\n\n" + piece).strip()
            else:
                if current:
                    chunks.append(current)
                current = piece
    if current:
        chunks.append(current)
    return chunks


def get_embedding(embed_model: TextEmbeddingModel, text: str) -> List[float]:
    resp = embed_model.get_embeddings([text])
    return resp[0].values  # list[float]


def store_document(embed_model: TextEmbeddingModel, filename: str, text: str) -> int:
    """Chunk a document, embed chunks and insert into pgvector table.
       Returns number of chunks inserted.
    """
    chunks = chunk_text(text)
    if not chunks:
        return 0

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            for idx, chunk in enumerate(chunks):
                emb = get_embedding(embed_model, chunk)
                cur.execute(
                    """
                    INSERT INTO documents (source_uri, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s::vector)
                    """,
                    (filename, idx, chunk, emb),
                )
        conn.commit()

    return len(chunks)


def search_similar_chunks(
    embed_model: TextEmbeddingModel,
    query: str,
    top_k: int = 5,
) -> List[Tuple[str, int, str]]:
    """Return (source_uri, chunk_index, content) for top_k similar chunks."""
    query_emb = get_embedding(embed_model, query)

    sql = """
    SELECT source_uri, chunk_index, content
    FROM documents
    ORDER BY embedding <-> %s::vector
    LIMIT %s;
    """

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (query_emb, top_k))
            rows = cur.fetchall()

    return [(r[0], r[1], r[2]) for r in rows]


def answer_question(qa_model: GenerativeModel, question: str, context: str) -> str:
    prompt = f"""
You are a helpful assistant. Use ONLY the following context to answer the question.
If the answer is not in the context, say you don't know and suggest the user upload more relevant documents.

Context:
{context}

Question:
{question}

Answer:
"""
    resp = qa_model.generate_content(prompt)
    return resp.text


# ---------- FastAPI app ----------

app = FastAPI()

# Initialise models & schema once on startup
embed_model_global: TextEmbeddingModel | None = None
qa_model_global: GenerativeModel | None = None


@app.on_event("startup")
async def startup_event():
    global embed_model_global, qa_model_global
    ensure_schema()
    embed_model_global, qa_model_global = init_vertex()


@app.get("/", response_class=HTMLResponse)
async def index():
    # Very simple HTML UI
    html = """
    <html>
      <head>
        <title>RAG Demo – Upload & Ask</title>
        <style>
          body { font-family: system-ui, -apple-system, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
          h1 { margin-bottom: 0.5rem; }
          .card { border: 1px solid #ddd; border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: 1.5rem; }
          label { font-weight: 600; }
          textarea, input[type="text"] { width: 100%; padding: 0.5rem; margin-top: 0.3rem; border-radius: 4px; border: 1px solid #ccc; }
          button { padding: 0.5rem 1rem; border-radius: 6px; border: none; background: #2563eb; color: white; font-weight: 600; cursor: pointer; }
          button:hover { background: #1d4ed8; }
          .sources { font-size: 0.9rem; color: #555; margin-top: 0.5rem; }
          pre { background: #f9fafb; padding: 0.75rem; border-radius: 6px; white-space: pre-wrap; }
        </style>
      </head>
      <body>
        <h1>RAG Demo – Upload & Ask</h1>
        <p>Upload one or more text files, then ask questions about them.</p>

        <div class="card">
          <h2>1. Upload documents</h2>
          <form action="/upload" method="post" enctype="multipart/form-data">
            <label for="files">Choose .txt / .md files:</label><br/>
            <input type="file" id="files" name="files" multiple />
            <br/><br/>
            <button type="submit">Upload & Embed</button>
          </form>
        </div>

        <div class="card">
          <h2>2. Ask a question</h2>
          <form action="/ask" method="post">
            <label for="question">Your question:</label><br/>
            <textarea id="question" name="question" rows="4" placeholder="e.g. What are the main responsibilities described in these documents?"></textarea>
            <br/><br/>
            <button type="submit">Ask</button>
          </form>
        </div>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.post("/upload", response_class=HTMLResponse)
async def upload(files: List[UploadFile] = File(...)):
    if embed_model_global is None:
        return HTMLResponse("Embedding model not initialised.", status_code=500)

    total_chunks = 0
    uploaded_names: List[str] = []

    for f in files:
        data = await f.read()
        try:
            text = data.decode("utf-8", errors="ignore")
        except Exception:
            continue
        chunks_added = store_document(embed_model_global, f.filename, text)
        total_chunks += chunks_added
        if chunks_added > 0:
            uploaded_names.append(f.filename)

    file_list = "<br>".join(uploaded_names) if uploaded_names else "None"

    html = f"""
    <html>
      <head><title>Upload complete</title></head>
      <body>
        <h1>Upload complete</h1>
        <p>Inserted <strong>{total_chunks}</strong> chunks into pgvector.</p>
        <p>Files processed:</p>
        <p>{file_list}</p>
        <p><a href="/">Back</a></p>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.post("/ask", response_class=HTMLResponse)
async def ask(question: str = Form(...)):
    if embed_model_global is None or qa_model_global is None:
        return HTMLResponse("Models not initialised.", status_code=500)

    # Fetch similar chunks
    chunks = search_similar_chunks(embed_model_global, question, top_k=5)

    if not chunks:
        html = """
        <html><body>
        <h1>No data available</h1>
        <p>The documents table is empty. Please upload some documents first.</p>
        <p><a href="/">Back</a></p>
        </body></html>
        """
        return HTMLResponse(content=html)

    # Build context + sources
    context_parts = []
    sources = []
    for src, idx, content in chunks:
        sources.append(f"{src} (chunk {idx})")
        context_parts.append(f"[{src} – chunk {idx}]\n{content}")
    context = "\n\n".join(context_parts)

    answer = answer_question(qa_model_global, question, context)

    sources_html = "<br>".join(sources)

    html = f"""
    <html>
      <head>
        <title>Answer</title>
      </head>
      <body>
        <h1>Answer</h1>
        <p><strong>Question:</strong> {question}</p>
        <h2>Response</h2>
        <pre>{answer}</pre>

        <div class="sources">
          <h3>Sources used:</h3>
          <p>{sources_html}</p>
        </div>

        <p><a href="/">Ask another question</a></p>
      </body>
    </html>
    """
    return HTMLResponse(content=html)
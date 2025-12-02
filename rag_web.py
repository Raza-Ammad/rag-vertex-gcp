import os
import io
import uuid
from typing import List, Optional, Dict, Any

import psycopg2
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

import PyPDF2

import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel


# --------------------------------------------------------------------
# Config (must be set in environment)
# --------------------------------------------------------------------

GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")

DB_HOST = os.environ.get("DB_HOST", "127.0.0.1")
DB_PORT = int(os.environ.get("DB_PORT", "5432"))
DB_NAME = os.environ.get("DB_NAME", "ragdb")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")

EMBEDDING_MODEL_NAME = "text-embedding-004"
GENERATION_MODEL_NAME = "gemini-2.0-flash"

if not GCP_PROJECT:
    raise RuntimeError("GCP_PROJECT environment variable is not set.")


# --------------------------------------------------------------------
# DB helpers
# --------------------------------------------------------------------

def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def ensure_schema() -> None:
    """Ensure the documents table exists.

    IMPORTANT: This matches the schema already used by ingest_pg.py:
      id SERIAL PRIMARY KEY
      source_uri TEXT
      chunk_index INT
      content TEXT
      embedding DOUBLE PRECISION[]
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    source_uri TEXT NOT NULL,
                    chunk_index INT NOT NULL,
                    content TEXT NOT NULL,
                    embedding DOUBLE PRECISION[]
                );
                """
            )
        conn.commit()


def list_documents() -> List[Dict[str, Any]]:
    """Return one row per source_uri, latest first, using only existing columns."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    source_uri,
                    split_part(source_uri, '::', 1) AS display_name,
                    COUNT(*) AS chunks,
                    MIN(id) AS first_id,
                    MAX(id) AS last_id
                FROM documents
                GROUP BY source_uri
                ORDER BY last_id DESC;
                """
            )
            rows = cur.fetchall()

    docs: List[Dict[str, Any]] = []
    for source_uri, display_name, chunks, first_id, last_id in rows:
        docs.append(
            {
                "source_uri": source_uri,
                "display_name": display_name,
                "chunks": chunks,
                "first_id": first_id,
                "last_id": last_id,
            }
        )
    return docs


# --------------------------------------------------------------------
# Vertex AI helpers
# --------------------------------------------------------------------

_embed_model: Optional[TextEmbeddingModel] = None
_qa_model: Optional[GenerativeModel] = None


def init_vertex():
    """Lazy init of embedding + QA models."""
    global _embed_model, _qa_model
    if _embed_model is None or _qa_model is None:
        print("🔧 Initialising Vertex AI (embeddings + Gemini)...")
        vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
        _embed_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
        _qa_model = GenerativeModel(GENERATION_MODEL_NAME)
    return _embed_model, _qa_model


def get_embedding(model: TextEmbeddingModel, text: str) -> List[float]:
    embeddings = model.get_embeddings([text])
    # vertexai returns an object whose .values is the vector
    return list(embeddings[0].values)


def _normalize_embedding(db_value) -> List[float]:
    """Turn Postgres stored embedding into a Python list[float]."""
    if db_value is None:
        return []
    if isinstance(db_value, list):
        return [float(x) for x in db_value]
    if isinstance(db_value, str):
        s = db_value.strip().strip("[]")
        if not s:
            return []
        return [float(x) for x in s.split(",")]
    return []


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def search_similar_chunks(
    embed_model: TextEmbeddingModel,
    query: str,
    top_k: int = 5,
    source_uri: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve top_k most similar chunks (cosine similarity in Python)."""
    query_emb = get_embedding(embed_model, query)

    with get_connection() as conn:
        with conn.cursor() as cur:
            if source_uri and source_uri != "all":
                cur.execute(
                    """
                    SELECT id, source_uri, chunk_index, content, embedding
                    FROM documents
                    WHERE source_uri = %s;
                    """,
                    (source_uri,),
                )
            else:
                cur.execute(
                    """
                    SELECT id, source_uri, chunk_index, content, embedding
                    FROM documents;
                    """
                )
            rows = cur.fetchall()

    scored: List[Dict[str, Any]] = []
    for row in rows:
        emb = _normalize_embedding(row[4])
        if not emb:
            continue
        score = cosine_similarity(query_emb, emb)
        scored.append(
            {
                "id": row[0],
                "source_uri": row[1],
                "chunk_index": row[2],
                "content": row[3],
                "score": score,
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def answer_question(
    qa_model: GenerativeModel, question: str, context: str
) -> str:
    prompt = f"""
You are a helpful assistant for a Retrieval-Augmented Generation (RAG) app.

Use ONLY the context below to answer the user's question.
If the answer is not in the context, say you don't know instead of guessing.

Context:
{context}

Question: {question}

Answer clearly and concisely.
"""
    resp = qa_model.generate_content(prompt)
    return resp.text.strip()


# --------------------------------------------------------------------
# Text / PDF handling
# --------------------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    pieces: List[str] = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        pieces.append(txt)
    return "\n".join(pieces)


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100,
) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


# --------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------

app = FastAPI(title="RAG Demo – Vertex AI + Postgres")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    ensure_schema()
    init_vertex()
    print("✅ Schema ensured and Vertex AI initialised.")


# --------------------------------------------------------------------
# HTML rendering
# --------------------------------------------------------------------

def render_home(
    docs: List[Dict[str, Any]],
    selected_source_uri: Optional[str] = None,
    question: str = "",
    answer: str = "",
    error: str = "",
) -> str:
    # Dropdown options
    options_html = '<option value="all">All documents</option>\n'
    for d in docs:
        selected_attr = "selected" if d["source_uri"] == selected_source_uri else ""
        label = f"{d['display_name']} ({d['chunks']} chunks)"
        options_html += (
            f'<option value="{d["source_uri"]}" {selected_attr}>{label}</option>\n'
        )

    # History table rows
    rows_html = ""
    for d in docs:
        rows_html += f"""
        <tr class="border-b border-gray-200/40">
          <td class="px-3 py-2 text-sm text-gray-100">{d['display_name']}</td>
          <td class="px-3 py-2 text-sm text-gray-400 text-center">{d['chunks']}</td>
          <td class="px-3 py-2 text-xs text-gray-500">#{d['last_id']}</td>
        </tr>
        """

    if not rows_html:
        rows_html = """
        <tr>
          <td colspan="3" class="px-3 py-4 text-center text-xs text-gray-500">
            No documents uploaded yet.
          </td>
        </tr>
        """

    error_block = ""
    if error:
        error_block = f"""
        <div class="mb-4 rounded-xl bg-red-50/10 border border-red-500/40 px-4 py-3 text-sm text-red-200">
          {error}
        </div>
        """

    answer_block = ""
    if answer:
        answer_block = f"""
        <div class="mt-4 rounded-2xl border border-slate-700 bg-slate-900/70 p-4 text-sm text-gray-100 max-h-72 overflow-y-auto">
          <div class="font-semibold mb-2 text-gray-50">Answer</div>
          <p class="whitespace-pre-line">{answer}</p>
        </div>
        """

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>RAG Demo</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="min-h-screen bg-slate-950 text-gray-100">
  <div class="max-w-6xl mx-auto px-4 py-8">
    <header class="mb-8 flex items-center justify-between">
      <div>
        <h1 class="text-2xl font-bold text-white">🧠 RAG Demo – Vertex AI + Postgres</h1>
        <p class="text-sm text-gray-400 mt-1">
          Upload a document (PDF or text), then ask questions grounded only in its content.
        </p>
      </div>
    </header>

    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
      <!-- Left: upload + history -->
      <section class="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-lg">
        <h2 class="text-lg font-semibold text-white mb-3">1. Upload document</h2>
        <form action="/upload" method="post" enctype="multipart/form-data" class="space-y-3">
          <input
            name="file"
            type="file"
            accept=".txt,.md,.pdf"
            class="block w-full text-sm text-gray-300
                   file:mr-4 file:py-2 file:px-4
                   file:rounded-full file:border-0
                   file:text-sm file:font-semibold
                   file:bg-indigo-500 file:text-white
                   hover:file:bg-indigo-600"
            required
          />
          <p class="text-xs text-gray-500">
            Supported: <span class="font-mono text-gray-300">.txt</span>,
            <span class="font-mono text-gray-300">.md</span>,
            <span class="font-mono text-gray-300">.pdf</span>
          </p>
          <button
            type="submit"
            class="inline-flex items-center px-4 py-2 rounded-full bg-indigo-500 hover:bg-indigo-600 text-sm font-medium text-white shadow-sm"
          >
            ⬆️ Upload &amp; index
          </button>
        </form>

        <div class="mt-6">
          <h3 class="text-sm font-semibold text-gray-200 mb-2">📂 Document history</h3>
          <div class="border border-slate-800 rounded-xl overflow-hidden bg-slate-950/60">
            <table class="min-w-full text-left text-xs">
              <thead class="bg-slate-900/90 text-gray-400">
                <tr>
                  <th class="px-3 py-2 font-medium">Name</th>
                  <th class="px-3 py-2 font-medium text-center">Chunks</th>
                  <th class="px-3 py-2 font-medium">Last ID</th>
                </tr>
              </thead>
              <tbody>
                {rows_html}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <!-- Right: ask + answer -->
      <section class="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-lg">
        <h2 class="text-lg font-semibold text-white mb-3">2. Ask a question</h2>
        {error_block}
        <form action="/ask" method="post" class="space-y-3">
          <div>
            <label class="block text-xs font-medium text-gray-300 mb-1">
              Choose document
            </label>
            <select
              name="source_uri"
              class="w-full rounded-xl border border-slate-700 bg-slate-950/70 text-sm px-3 py-2 text-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              {options_html}
            </select>
          </div>

          <div>
            <label class="block text-xs font-medium text-gray-300 mb-1 mt-3">
              Your question
            </label>
            <textarea
              name="question"
              rows="3"
              class="w-full rounded-xl border border-slate-700 bg-slate-950/70 text-sm px-3 py-2 text-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              placeholder="e.g. What is this document mainly about?"
            >{question}</textarea>
          </div>

          <button
            type="submit"
            class="inline-flex items-center px-4 py-2 rounded-full bg-emerald-500 hover:bg-emerald-600 text-sm font-medium text-white shadow-sm"
          >
            💬 Ask
          </button>
        </form>

        {answer_block}
      </section>
    </div>
  </div>
</body>
</html>
"""
    return html


# --------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    docs = list_documents()
    return HTMLResponse(render_home(docs))


@app.post("/upload", response_class=HTMLResponse)
async def upload(file: UploadFile = File(...)):
    docs_before = list_documents()
    embed_model, _ = init_vertex()

    try:
        contents = await file.read()
        filename = file.filename or "uploaded_document"

        # Extract text depending on extension
        if filename.lower().endswith(".pdf"):
            text = extract_text_from_pdf(contents)
        else:
            text = contents.decode("utf-8", errors="ignore")

        if not text.strip():
            return HTMLResponse(
                render_home(
                    docs_before,
                    error="The uploaded file appears to be empty.",
                ),
                status_code=400,
            )

        chunks = chunk_text(text)
        if not chunks:
            return HTMLResponse(
                render_home(
                    docs_before,
                    error="Could not split the document into chunks.",
                ),
                status_code=400,
            )

        # Use filename + random suffix so re-uploads are unique
        source_uri = f"{filename}::{uuid.uuid4().hex[:8]}"

        with get_connection() as conn:
            with conn.cursor() as cur:
                for idx, chunk in enumerate(chunks):
                    emb = get_embedding(embed_model, chunk)
                    cur.execute(
                        """
                        INSERT INTO documents (source_uri, chunk_index, content, embedding)
                        VALUES (%s, %s, %s, %s);
                        """,
                        (source_uri, idx, chunk, emb),
                    )
            conn.commit()

        docs_after = list_documents()
        msg = (
            f"Uploaded '{filename}' as {len(chunks)} chunks. "
            "You can now ask questions about it."
        )
        return HTMLResponse(
            render_home(
                docs_after,
                selected_source_uri=source_uri,
                answer=msg,
            )
        )

    except Exception as e:
        docs_now = list_documents()
        return HTMLResponse(
            render_home(
                docs_now,
                error=f"Error while uploading: {e}",
            ),
            status_code=500,
        )


@app.post("/ask", response_class=HTMLResponse)
async def ask(
    question: str = Form(""),
    source_uri: str = Form("all"),
):
    docs = list_documents()
    embed_model, qa_model = init_vertex()

    question = (question or "").strip()
    if not question:
        return HTMLResponse(
            render_home(
                docs,
                selected_source_uri=source_uri,
                error="Please enter a question.",
            ),
            status_code=400,
        )

    try:
        chunks = search_similar_chunks(
            embed_model,
            question,
            top_k=5,
            source_uri=None if source_uri == "all" else source_uri,
        )

        if not chunks:
            answer = (
                "I couldn't find any indexed content for that selection. "
                "Try uploading a document or choose 'All documents'."
            )
        else:
            context_text = "\n\n---\n\n".join(c["content"] for c in chunks)
            answer = answer_question(qa_model, question, context_text)

        return HTMLResponse(
            render_home(
                docs,
                selected_source_uri=source_uri,
                question=question,
                answer=answer,
            )
        )

    except Exception as e:
        return HTMLResponse(
            render_home(
                docs,
                selected_source_uri=source_uri,
                question=question,
                error=f"Error while answering: {e}",
            ),
            status_code=500,
        )
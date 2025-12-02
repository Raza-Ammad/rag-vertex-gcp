import os
import io
from typing import List, Optional, Dict, Any

import psycopg2
import psycopg2.extras

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel

# ---- OPTIONAL: if you already had a PDF lib working, keep using it.
# Here we use PyPDF2 which is very common. If you already use another
# library (like pypdf or pdfplumber) you can swap this part easily.
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None  # We will error nicely if PDF is uploaded


# ------------------------
# Config: GCP + Database
# ------------------------

GCP_PROJECT = os.environ.get("GCP_PROJECT", "starry-journal-480011-m8")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")

DB_HOST = os.environ.get("DB_HOST", "35.184.43.16")
DB_PORT = int(os.environ.get("DB_PORT", "5432"))
DB_NAME = os.environ.get("DB_NAME", "ragdb")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "Bhatti@512")

EMBEDDING_MODEL_NAME = "text-embedding-004"
GENERATION_MODEL_NAME = "gemini-2.0-flash"

# Globals for Vertex models
_embed_model: Optional[TextEmbeddingModel] = None
_qa_model: Optional[GenerativeModel] = None

app = FastAPI(title="RAG Demo (Vertex + Cloud SQL)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------
# DB + Vertex helpers
# ------------------------

def get_connection():
    """Plain psycopg2 connection to Cloud SQL (public IP)."""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def ensure_table_exists():
    """Create documents table if it doesn't exist (schema matches ingest_pg.py)."""
    create_sql = """
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        source_uri TEXT NOT NULL,
        chunk_index INT NOT NULL,
        content TEXT NOT NULL,
        embedding DOUBLE PRECISION[] NOT NULL
    );
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(create_sql)
        conn.commit()


def init_vertex():
    """Initialise Vertex AI models once (same style as rag_query.py)."""
    global _embed_model, _qa_model

    if _embed_model is not None and _qa_model is not None:
        return _embed_model, _qa_model

    vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
    _embed_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
    _qa_model = GenerativeModel(GENERATION_MODEL_NAME)

    return _embed_model, _qa_model


def get_embedding(model: TextEmbeddingModel, text: str) -> List[float]:
    """Get a single embedding vector for text."""
    resp = model.get_embeddings([text])
    return resp[0].values


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    import math

    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """Simple overlap-based chunking by characters."""
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return chunks


def store_document(source_uri: str, full_text: str):
    """Chunk, embed and store a new document into Postgres."""
    embed_model, _ = init_vertex()
    chunks = chunk_text(full_text)

    if not chunks:
        return

    with get_connection() as conn:
        with conn.cursor() as cur:
            for idx, chunk in enumerate(chunks):
                emb = get_embedding(embed_model, chunk)
                cur.execute(
                    """
                    INSERT INTO documents (source_uri, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (source_uri, idx, chunk, emb),
                )
        conn.commit()


def list_documents() -> List[Dict[str, Any]]:
    """
    Return a list of logical documents, grouped by source_uri.
    We don't depend on uploaded_at — we just order by MIN(id) (latest last).
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    source_uri,
                    COUNT(*) AS chunk_count,
                    MIN(id) AS first_id
                FROM documents
                GROUP BY source_uri
                ORDER BY MIN(id) DESC;
                """
            )
            rows = cur.fetchall()

    docs = []
    for source_uri, chunk_count, first_id in rows:
        docs.append(
            {
                "source_uri": source_uri,
                "chunk_count": chunk_count,
                "first_id": first_id,
            }
        )
    return docs


def search_similar_chunks(
    query: str,
    top_k: int = 5,
    source_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Embed the query and find top-k similar chunks in Python (no pgvector)."""
    embed_model, _ = init_vertex()
    query_emb = get_embedding(embed_model, query)

    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if source_filter and source_filter != "ALL":
                cur.execute(
                    """
                    SELECT id, source_uri, chunk_index, content, embedding
                    FROM documents
                    WHERE source_uri = %s
                    """,
                    (source_filter,),
                )
            else:
                cur.execute(
                    """
                    SELECT id, source_uri, chunk_index, content, embedding
                    FROM documents
                    """
                )
            rows = cur.fetchall()

    scored = []
    for row in rows:
        emb = row["embedding"]
        sim = cosine_similarity(query_emb, emb)
        scored.append(
            {
                "id": row["id"],
                "source_uri": row["source_uri"],
                "chunk_index": row["chunk_index"],
                "content": row["content"],
                "similarity": sim,
            }
        )

    scored.sort(key=lambda r: r["similarity"], reverse=True)
    return scored[:top_k]


def answer_question(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    """Call Gemini with context + question and return plain text answer."""
    _, qa_model = init_vertex()

    context_text = "\n\n".join(
        f"[chunk {c['chunk_index']} from {c['source_uri']}]\n{c['content']}"
        for c in context_chunks
    )

    prompt = f"""You are a helpful assistant. Use ONLY the context below to answer the user's question.

CONTEXT:
{context_text}

QUESTION:
{question}

If the context is not enough to answer reliably, say so clearly.
"""

    resp = qa_model.generate_content(prompt)
    return resp.text.strip() if resp and getattr(resp, "text", None) else "(no answer)"


# ------------------------
# FastAPI lifecycle
# ------------------------

@app.on_event("startup")
def on_startup():
    # Make sure table exists and Vertex is ready
    ensure_table_exists()
    init_vertex()


# ------------------------
# HTML UI rendering
# ------------------------

def render_page(
    docs: List[Dict[str, Any]],
    message: str = "",
    question: str = "",
    answer: str = "",
    selected_source: str = "ALL",
) -> HTMLResponse:
    def escape_html(s: str) -> str:
        import html
        return html.escape(s, quote=True)

    docs_html = ""
    if docs:
        docs_html += '<div class="doc-list">'
        docs_html += '<label class="doc-option">'
        docs_html += f'<input type="radio" name="doc_source" value="ALL" {"checked" if selected_source == "ALL" else ""}>'
        docs_html += " All documents"
        docs_html += "</label>"
        for d in docs:
            src = d["source_uri"]
            docs_html += '<label class="doc-option">'
            docs_html += (
                f'<input type="radio" name="doc_source" value="{escape_html(src)}" '
            )
            if selected_source == src:
                docs_html += "checked"
            docs_html += ">"
            docs_html += f"{escape_html(src)} "
            docs_html += f'<span class="doc-meta">({d["chunk_count"]} chunks)</span>'
            docs_html += "</label>"
        docs_html += "</div>"
    else:
        docs_html = "<p>No documents uploaded yet.</p>"

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>RAG Demo – Vertex + Cloud SQL</title>
    <style>
        body {{
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #0f172a;
            color: #e5e7eb;
            margin: 0;
            padding: 0;
        }}
        .page {{
            max-width: 1100px;
            margin: 0 auto;
            padding: 24px;
        }}
        h1 {{
            font-size: 26px;
            margin-bottom: 4px;
        }}
        .subtitle {{
            color: #9ca3af;
            font-size: 14px;
            margin-bottom: 20px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1.1fr 1.4fr;
            gap: 20px;
        }}
        .card {{
            background: #020617;
            border-radius: 12px;
            padding: 16px 18px;
            border: 1px solid #1e293b;
            box-shadow: 0 10px 30px rgba(15,23,42,0.6);
        }}
        .card h2 {{
            font-size: 18px;
            margin-top: 0;
            margin-bottom: 10px;
        }}
        label {{
            font-size: 14px;
            display: block;
            margin-bottom: 4px;
            color: #cbd5f5;
        }}
        input[type="file"],
        input[type="text"],
        textarea,
        select {{
            width: 100%;
            padding: 8px 10px;
            border-radius: 8px;
            border: 1px solid #1e293b;
            background: #020617;
            color: #e5e7eb;
            font-size: 14px;
            box-sizing: border-box;
        }}
        textarea {{
            min-height: 120px;
            resize: vertical;
        }}
        .btn {{
            display: inline-block;
            margin-top: 8px;
            padding: 8px 14px;
            border-radius: 999px;
            border: none;
            background: linear-gradient(135deg, #22c55e, #16a34a);
            color: #020617;
            font-weight: 600;
            cursor: pointer;
            font-size: 14px;
        }}
        .btn:hover {{
            filter: brightness(1.08);
        }}
        .message {{
            margin-top: 8px;
            font-size: 13px;
            color: #a5b4fc;
        }}
        .doc-list {{
            margin-top: 6px;
            max-height: 230px;
            overflow-y: auto;
            border: 1px solid #1f2937;
            border-radius: 10px;
            padding: 10px;
            background: #020617;
        }}
        .doc-option {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 13px;
            padding: 4px 2px;
            cursor: pointer;
        }}
        .doc-option input {{
            margin-right: 6px;
        }}
        .doc-meta {{
            color: #6b7280;
            font-size: 11px;
        }}
        .answer-block {{
            margin-top: 10px;
            padding: 10px 12px;
            border-radius: 10px;
            background: #020617;
            border: 1px solid #1e293b;
            font-size: 14px;
            white-space: pre-wrap;
        }}
        .question-label {{
            display: flex;
            justify-content: space-between;
            font-size: 13px;
            margin-bottom: 4px;
        }}
        .tag {{
            font-size: 11px;
            padding: 2px 8px;
            border-radius: 999px;
            border: 1px solid #374151;
            color: #9ca3af;
        }}
    </style>
</head>
<body>
    <div class="page">
        <h1>RAG Demo – Vertex AI + Cloud SQL</h1>
        <div class="subtitle">
            Upload a document (PDF or text), then ask questions grounded only in your stored chunks.
        </div>

        <div class="grid">
            <!-- Left: upload + history -->
            <div class="card">
                <h2>1. Upload document</h2>
                <form action="/upload" method="post" enctype="multipart/form-data">
                    <label for="file">Choose file (PDF or .txt)</label>
                    <input type="file" id="file" name="file" required />

                    <label for="custom_name" style="margin-top:8px;">Optional name override</label>
                    <input type="text" id="custom_name" name="custom_name" placeholder="If blank, we use the file name" />

                    <button type="submit" class="btn">Upload & Embed</button>
                </form>

                {"<div class='message'>" + escape_html(message) + "</div>" if message else ""}

                <h2 style="margin-top:18px;">2. Document history</h2>
                <div style="font-size:12px;color:#9ca3af;margin-bottom:4px;">
                    Select which document you want to query.
                </div>
                <form id="doc-select-form">
                    {docs_html}
                </form>
            </div>

            <!-- Right: Ask questions -->
            <div class="card">
                <h2>3. Ask a question</h2>
                <form action="/ask" method="post">
                    <div class="question-label">
                        <span>Your question</span>
                        <span class="tag">Context: selected document or all</span>
                    </div>
                    <textarea name="question" placeholder="Ask something about your documents...">{escape_html(question)}</textarea>

                    <!-- This will be set by JS to the selected doc_source -->
                    <input type="hidden" name="doc_source" id="doc_source_hidden" value="{escape_html(selected_source)}" />

                    <button type="submit" class="btn">Ask</button>
                </form>

                {"<div class='answer-block'><strong>Answer:</strong> " + escape_html(answer) + "</div>" if answer else ""}
            </div>
        </div>
    </div>

    <script>
        // Sync selected radio -> hidden input so /ask knows which doc to use
        function syncSelectedDoc() {{
            const radios = document.querySelectorAll('input[name="doc_source"]');
            let selected = "ALL";
            for (const r of radios) {{
                if (r.checked) {{
                    selected = r.value;
                }}
            }}
            const hidden = document.getElementById("doc_source_hidden");
            if (hidden) {{
                hidden.value = selected;
            }}
        }}

        document.addEventListener("change", function(e) {{
            if (e.target && e.target.name === "doc_source") {{
                syncSelectedDoc();
            }}
        }});

        // Initial sync on load
        syncSelectedDoc();
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html)


# ------------------------
# Routes
# ------------------------

@app.get("/", response_class=HTMLResponse)
def home():
    docs = list_documents()
    return render_page(docs)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), custom_name: str = Form("")):
    filename = custom_name.strip() or file.filename or "uploaded_document"

    content_type = (file.content_type or "").lower()
    raw_bytes = await file.read()

    text = ""
    if content_type == "application/pdf" or filename.lower().endswith(".pdf"):
        if PyPDF2 is None:
            return HTMLResponse(
                "PDF support not installed on server (PyPDF2 missing).",
                status_code=500,
            )
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(raw_bytes))  # type: ignore
            pages = [page.extract_text() or "" for page in reader.pages]
            text = "\n\n".join(pages)
        except Exception as e:
            return HTMLResponse(f"Error reading PDF: {e}", status_code=500)
    else:
        # Assume text
        try:
            text = raw_bytes.decode("utf-8", errors="ignore")
        except Exception as e:
            return HTMLResponse(f"Error decoding file as text: {e}", status_code=500)

    text = text.strip()
    if not text:
        docs = list_documents()
        return render_page(docs, message="Uploaded file contained no extractable text.")

    # Embed + store
    store_document(filename, text)
    docs = list_documents()
    return render_page(docs, message=f"Uploaded and embedded '{filename}' successfully.")


@app.post("/ask", response_class=HTMLResponse)
async def ask(question: str = Form(""), doc_source: str = Form("ALL")):
    question = (question or "").strip()
    if not question:
        docs = list_documents()
        return render_page(docs, message="Please enter a question.", selected_source=doc_source)

    chunks = search_similar_chunks(question, top_k=5, source_filter=doc_source)
    if not chunks:
        docs = list_documents()
        return render_page(
            docs,
            message="No chunks found in the database yet. Upload a document first.",
            question=question,
            selected_source=doc_source,
        )

    answer = answer_question(question, chunks)
    docs = list_documents()
    return render_page(
        docs,
        question=question,
        answer=answer,
        selected_source=doc_source,
    )


@app.get("/api/documents")
def api_documents():
    """Small JSON endpoint listing doc history (for debugging / future use)."""
    return JSONResponse(list_documents())
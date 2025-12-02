import os
import io
import html
from typing import List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from pypdf import PdfReader

import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel

# ---- Config from environment ----
PROJECT_ID = os.environ.get("GCP_PROJECT", "starry-journal-480011-m8")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")

DB_HOST = os.environ.get("DB_HOST", "35.184.43.16")
DB_PORT = int(os.environ.get("DB_PORT", "5432"))
DB_NAME = os.environ.get("DB_NAME", "ragdb")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "postgres")

EMBEDDING_MODEL_NAME = "text-embedding-004"
GENERATION_MODEL_NAME = "gemini-2.0-flash"

# ---- DB helpers ----


def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        connect_timeout=10,
    )


def ensure_schema():
    ddl = """
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        source_uri TEXT NOT NULL,
        chunk_index INT NOT NULL,
        content TEXT NOT NULL,
        embedding VECTOR(768),
        uploaded_at TIMESTAMPTZ DEFAULT NOW()
    );
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()


def list_documents(limit: int = 100) -> List[str]:
    """Return distinct document names (source_uri) for history / dropdown."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT source_uri
                FROM documents
                GROUP BY source_uri
                ORDER BY MAX(uploaded_at) DESC
                LIMIT %s;
                """,
                (limit,),
            )
            rows = cur.fetchall()
    return [r[0] for r in rows]


# ---- Vertex AI helpers ----


def init_vertex():
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    embed_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
    qa_model = GenerativeModel(GENERATION_MODEL_NAME)
    return embed_model, qa_model


def get_embedding(model: TextEmbeddingModel, text: str) -> List[float]:
    response = model.get_embeddings([text])
    return response[0].values


# ---- Text & PDF processing ----


def extract_text_from_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages.append(page_text)
    return "\n".join(pages)


def chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> List[str]:
    text = text.replace("\r", "")
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return chunks


def insert_document_chunks(
    embed_model: TextEmbeddingModel, source_uri: str, text: str
) -> int:
    chunks = chunk_text(text)
    if not chunks:
        return 0

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
    return len(chunks)


def search_similar_chunks(
    embed_model: TextEmbeddingModel,
    query: str,
    top_k: int = 5,
    source_filter: Optional[str] = None,
):
    query_emb = get_embedding(embed_model, query)

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if source_filter:
                cur.execute(
                    """
                    SELECT
                        id,
                        source_uri,
                        chunk_index,
                        content,
                        1 - (embedding <#> %s::vector) AS similarity
                    FROM documents
                    WHERE source_uri = %s
                    ORDER BY embedding <#> %s::vector
                    LIMIT %s;
                    """,
                    (query_emb, source_filter, query_emb, top_k),
                )
            else:
                cur.execute(
                    """
                    SELECT
                        id,
                        source_uri,
                        chunk_index,
                        content,
                        1 - (embedding <#> %s::vector) AS similarity
                    FROM documents
                    ORDER BY embedding <#> %s::vector
                    LIMIT %s;
                    """,
                    (query_emb, query_emb, top_k),
                )
            rows = cur.fetchall()
    return rows


def answer_question(
    qa_model: GenerativeModel, question: str, context_chunks: List[dict]
) -> str:
    context_text = "\n\n".join(
        f"[{c['source_uri']} #{c['chunk_index']}] {c['content']}"
        for c in context_chunks
    )
    prompt = f"""
You are a helpful assistant answering questions using ONLY the context below.

Context:
{context_text}

Question:
{question}

If the answer is not clearly supported by the context, say
"I’m not sure – the document doesn’t contain enough information."
"""
    resp = qa_model.generate_content(prompt)
    return resp.text


# ---- FastAPI app ----

app = FastAPI(title="RAG Demo – Vertex AI + Cloud SQL")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Init at startup
ensure_schema()
EMBED_MODEL, QA_MODEL = init_vertex()


def render_home(message: str = "") -> HTMLResponse:
    docs = list_documents()
    options_html = "\n".join(
        f'<option value="{html.escape(d)}">{html.escape(d)}</option>' for d in docs
    )

    html_page = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>RAG Demo – Vertex AI + Cloud SQL</title>
  <style>
    * {{
      box-sizing: border-box;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    body {{
      margin: 0;
      padding: 0;
      background: radial-gradient(circle at top left, #111827, #020617);
      color: #e5e7eb;
      min-height: 100vh;
      display: flex;
      justify-content: center;
      align-items: flex-start;
    }}
    .container {{
      width: 100%;
      max-width: 900px;
      margin: 40px 16px;
      background: rgba(15, 23, 42, 0.95);
      border-radius: 16px;
      padding: 24px 28px 32px;
      box-shadow: 0 18px 45px rgba(0, 0, 0, 0.55);
      border: 1px solid rgba(148, 163, 184, 0.3);
    }}
    h1 {{
      font-size: 1.6rem;
      margin-bottom: 4px;
    }}
    .subtitle {{
      font-size: 0.9rem;
      color: #9ca3af;
      margin-bottom: 18px;
    }}
    .section-title {{
      font-size: 1rem;
      margin: 20px 0 8px;
      color: #e5e7eb;
    }}
    .card {{
      border-radius: 12px;
      border: 1px solid rgba(55, 65, 81, 0.9);
      padding: 14px 16px 16px;
      background: linear-gradient(145deg, rgba(15, 23, 42, 0.9), rgba(17, 24, 39, 0.95));
      margin-bottom: 14px;
    }}
    label {{
      font-size: 0.9rem;
      color: #e5e7eb;
      display: block;
      margin-bottom: 6px;
    }}
    input[type="file"],
    select,
    textarea {{
      width: 100%;
      border-radius: 8px;
      border: 1px solid #374151;
      background: #020617;
      color: #e5e7eb;
      padding: 8px 10px;
      font-size: 0.9rem;
    }}
    textarea {{
      min-height: 70px;
      resize: vertical;
    }}
    button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 8px 14px;
      border-radius: 9999px;
      border: none;
      background: linear-gradient(135deg, #22c55e, #16a34a);
      color: #ecfdf5;
      font-weight: 500;
      font-size: 0.9rem;
      cursor: pointer;
      margin-top: 8px;
      box-shadow: 0 10px 25px rgba(34, 197, 94, 0.35);
      transition: transform 0.08s ease, box-shadow 0.08s ease, filter 0.08s ease;
    }}
    button:hover {{
      transform: translateY(-1px);
      filter: brightness(1.05);
      box-shadow: 0 14px 30px rgba(34, 197, 94, 0.45);
    }}
    button:active {{
      transform: translateY(0);
      box-shadow: 0 8px 18px rgba(34, 197, 94, 0.35);
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      padding: 4px 10px;
      border-radius: 999px;
      background: rgba(55, 65, 81, 0.9);
      font-size: 0.75rem;
      color: #e5e7eb;
      margin-right: 6px;
      margin-bottom: 6px;
    }}
    .pill span {{
      opacity: 0.75;
      margin-left: 4px;
      font-size: 0.7rem;
    }}
    .message {{
      font-size: 0.85rem;
      margin-bottom: 10px;
      color: #a5b4fc;
    }}
    .docs-list {{
      margin-top: 8px;
      max-height: 120px;
      overflow-y: auto;
      border-radius: 8px;
      border: 1px solid #1f2937;
      padding: 8px;
      background: rgba(15, 23, 42, 0.7);
      font-size: 0.8rem;
    }}
    .docs-list div {{
      padding: 3px 0;
      border-bottom: 1px solid rgba(31, 41, 55, 0.7);
    }}
    .docs-list div:last-child {{
      border-bottom: none;
    }}
    .docs-list span {{
      color: #9ca3af;
      font-size: 0.75rem;
    }}
    .footer {{
      margin-top: 18px;
      font-size: 0.75rem;
      color: #6b7280;
      text-align: right;
    }}
  </style>
</head>
<body>
  <main class="container">
    <header>
      <h1>RAG Demo – Vertex AI + Cloud SQL</h1>
      <p class="subtitle">Upload documents (TXT or PDF), then ask questions grounded in your own data.</p>
      {"<p class='message'>" + html.escape(message) + "</p>" if message else ""}
    </header>

    <section class="card">
      <h2 class="section-title">1. Upload documents</h2>
      <form action="/upload" method="post" enctype="multipart/form-data">
        <label for="files">Choose one or more files (TXT or PDF):</label>
        <input id="files" name="files" type="file" multiple required />
        <button type="submit">Upload &amp; Embed</button>
      </form>

      <h3 class="section-title" style="margin-top: 16px; font-size: 0.9rem;">Upload history</h3>
      {"<div class='docs-list'>" if docs else "<p class='docs-list'>No documents uploaded yet.</p>"}
      { "".join(f"<div>{html.escape(d)}</div>" for d in docs) if docs else "" }
      {"</div>" if docs else ""}
    </section>

    <section class="card">
      <h2 class="section-title">2. Ask a question</h2>
      <form action="/ask" method="post">
        <label for="doc">Limit search to a single document (optional):</label>
        <select id="doc" name="source_uri_filter">
          <option value="">All documents</option>
          {options_html}
        </select>

        <label for="question" style="margin-top: 10px;">Your question:</label>
        <textarea id="question" name="question" placeholder="e.g. What is the main topic of this document?" required></textarea>

        <button type="submit">Ask with RAG</button>
      </form>
    </section>

    <div class="footer">
      Vertex AI · Cloud SQL (pgvector) · FastAPI
    </div>
  </main>
</body>
</html>
"""
    return HTMLResponse(html_page)


@app.get("/", response_class=HTMLResponse)
def home():
    return render_home()


@app.post("/upload", response_class=HTMLResponse)
async def upload(files: List[UploadFile] = File(...)):
    total_chunks = 0
    uploaded_files = []

    for file in files:
        raw = await file.read()
        filename = file.filename or "uploaded_file"

        if filename.lower().endswith(".pdf") or (
            file.content_type and "pdf" in file.content_type.lower()
        ):
            text = extract_text_from_pdf(raw)
        else:
            text = raw.decode("utf-8", errors="ignore")

        if text.strip():
            chunks = insert_document_chunks(EMBED_MODEL, filename, text)
            total_chunks += chunks
            uploaded_files.append(filename)

    message = f"Uploaded {len(uploaded_files)} file(s), stored {total_chunks} chunk(s)."
    return render_home(message=message)


@app.post("/ask", response_class=HTMLResponse)
async def ask(
    question: str = Form(...),
    source_uri_filter: Optional[str] = Form(None),
):
    question = question.strip()
    if not question:
        return RedirectResponse("/", status_code=303)

    chunks = search_similar_chunks(
        EMBED_MODEL, question, top_k=5, source_filter=source_uri_filter or None
    )

    if not chunks:
        answer = "I couldn't find any data in the database yet. Try uploading a document first."
        context_text = ""
    else:
        answer = answer_question(QA_MODEL, question, chunks)
        context_lines = []
        for c in chunks:
            context_lines.append(
                f"- {c['source_uri']} (chunk #{c['chunk_index']}, similarity {c['similarity']:.3f})"
            )
        context_text = "\n".join(context_lines)

    docs = list_documents()
    options_html = "\n".join(
        f'<option value="{html.escape(d)}" {"selected" if d == (source_uri_filter or "") else ""}>{html.escape(d)}</option>'
        for d in docs
    )

    context_html = (
        "<ul>"
        + "".join(
            f"<li><strong>{html.escape(c['source_uri'])}</strong> – chunk #{c['chunk_index']} (sim {c['similarity']:.3f})</li>"
            for c in chunks
        )
        + "</ul>"
        if chunks
        else "<p>No context chunks retrieved.</p>"
    )

    page = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>RAG Answer</title>
  <style>
    * {{
      box-sizing: border-box;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    body {{
      margin: 0;
      padding: 0;
      background: radial-gradient(circle at top left, #111827, #020617);
      color: #e5e7eb;
      min-height: 100vh;
      display: flex;
      justify-content: center;
      align-items: flex-start;
    }}
    .container {{
      width: 100%;
      max-width: 900px;
      margin: 40px 16px;
      background: rgba(15, 23, 42, 0.95);
      border-radius: 16px;
      padding: 24px 28px 32px;
      box-shadow: 0 18px 45px rgba(0, 0, 0, 0.55);
      border: 1px solid rgba(148, 163, 184, 0.3);
    }}
    h1 {{
      font-size: 1.5rem;
      margin-bottom: 8px;
    }}
    .subtitle {{
      font-size: 0.9rem;
      color: #9ca3af;
      margin-bottom: 18px;
    }}
    .card {{
      border-radius: 12px;
      border: 1px solid rgba(55, 65, 81, 0.9);
      padding: 14px 16px 16px;
      background: linear-gradient(145deg, rgba(15, 23, 42, 0.9), rgba(17, 24, 39, 0.95));
      margin-bottom: 14px;
    }}
    pre {{
      white-space: pre-wrap;
      word-wrap: break-word;
      font-size: 0.9rem;
      background: #020617;
      padding: 10px 12px;
      border-radius: 8px;
      border: 1px solid #1f2937;
    }}
    a.button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 8px 14px;
      border-radius: 9999px;
      border: none;
      background: linear-gradient(135deg, #22c55e, #16a34a);
      color: #ecfdf5;
      font-weight: 500;
      font-size: 0.9rem;
      cursor: pointer;
      text-decoration: none;
      margin-top: 8px;
      box-shadow: 0 10px 25px rgba(34, 197, 94, 0.35);
    }}
    dl {{
      font-size: 0.9rem;
    }}
    dt {{
      font-weight: 600;
      margin-top: 8px;
    }}
    dd {{
      margin: 4px 0 0 0;
      color: #9ca3af;
    }}
  </style>
</head>
<body>
  <main class="container">
    <h1>RAG Answer</h1>
    <p class="subtitle">Question answered using pgvector + Vertex AI.</p>

    <section class="card">
      <h2 style="font-size: 1rem;">Question</h2>
      <pre>{html.escape(question)}</pre>
    </section>

    <section class="card">
      <h2 style="font-size: 1rem;">Answer</h2>
      <pre>{html.escape(answer)}</pre>
    </section>

    <section class="card">
      <h2 style="font-size: 1rem;">Sources used</h2>
      {context_html}
      <details style="margin-top: 8px;">
        <summary style="cursor:pointer; font-size:0.85rem; color:#9ca3af;">Raw context summary</summary>
        <pre>{html.escape(context_text)}</pre>
      </details>
    </section>

    <section class="card">
      <h2 style="font-size: 1rem;">Ask another question</h2>
      <form action="/ask" method="post">
        <label for="doc" style="font-size: 0.85rem;">Limit search to a single document (optional):</label>
        <select id="doc" name="source_uri_filter" style="width:100%; margin-bottom:8px; padding:6px 8px; border-radius:8px; border:1px solid #374151; background:#020617; color:#e5e7eb;">
          <option value="">All documents</option>
          {options_html}
        </select>

        <label for="question" style="font-size: 0.85rem;">Your question:</label>
        <textarea id="question" name="question" style="width:100%; min-height:70px; border-radius:8px; border:1px solid #374151; background:#020617; color:#e5e7eb; padding:8px 10px; margin-top:4px;" required></textarea>

        <button type="submit" style="margin-top:10px;">Ask again</button>
      </form>
    </section>

    <a href="/" class="button">⬅ Back to home</a>
  </main>
</body>
</html>
"""
    return HTMLResponse(page)
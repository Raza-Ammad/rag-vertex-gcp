import os
from io import BytesIO
from typing import List, Tuple

import psycopg2
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
from pypdf import PdfReader

# ---------------- Config ----------------

PROJECT_ID = os.getenv("GCP_PROJECT", "starry-journal-480011-m8")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")

DB_HOST = os.getenv("DB_HOST", "35.184.43.16")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "ragdb")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

EMBEDDING_MODEL_NAME = "text-embedding-004"
GENERATOR_MODEL_NAME = "gemini-2.0-flash"

_embed_model: TextEmbeddingModel | None = None
_qa_model: GenerativeModel | None = None

# ---------------- DB helpers ----------------


def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def init_db_schema() -> None:
    """Ensure pgvector extension + documents table exist."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            # pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            # basic documents table (same shape as used by ingest_pg / rag_query)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    source_uri TEXT,
                    chunk_index INT,
                    content TEXT,
                    embedding VECTOR(768)
                );
                """
            )
            conn.commit()


# ---------------- Vertex helpers ----------------


def init_vertex() -> Tuple[TextEmbeddingModel, GenerativeModel]:
    global _embed_model, _qa_model

    if _embed_model is None or _qa_model is None:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        _embed_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
        _qa_model = GenerativeModel(GENERATOR_MODEL_NAME)

    return _embed_model, _qa_model


def get_embedding(model: TextEmbeddingModel, text: str) -> List[float]:
    """Return a Python list of floats for a single text."""
    resp = model.get_embeddings([text])
    return list(resp[0].values)


def to_vector_literal(vec: List[float]) -> str:
    """Format a Python list as a Postgres vector literal."""
    return "[" + ",".join(str(v) for v in vec) + "]"


# ---------------- Text / PDF handling ----------------


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
    """Simple character-based chunking."""
    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + max_chars, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = end - overlap

    return chunks


def extract_text_from_pdf(data: bytes) -> str:
    """Extract plain text from a PDF file (all pages)."""
    reader = PdfReader(BytesIO(data))
    pages_text: List[str] = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages_text.append(page_text)

    return "\n\n".join(pages_text)


def store_document(source_uri: str, full_text: str) -> int:
    """Chunk + embed a document, store into Postgres, return number of chunks."""
    embed_model, _ = init_vertex()
    chunks = chunk_text(full_text)

    if not chunks:
        return 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Optional: delete previous chunks for same source_uri
            cur.execute("DELETE FROM documents WHERE source_uri = %s;", (source_uri,))

            for idx, chunk in enumerate(chunks):
                emb_list = get_embedding(embed_model, chunk)
                emb_literal = to_vector_literal(emb_list)
                cur.execute(
                    """
                    INSERT INTO documents (source_uri, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s::vector)
                    """,
                    (source_uri, idx, chunk, emb_literal),
                )

        conn.commit()

    return len(chunks)


def search_similar_chunks(query: str, top_k: int = 5) -> List[Tuple[int, str, int, str]]:
    """Return (id, source_uri, chunk_index, content) for the most similar chunks."""
    embed_model, _ = init_vertex()
    query_vec = get_embedding(embed_model, query)
    vec_lit = to_vector_literal(query_vec)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, source_uri, chunk_index, content
                FROM documents
                ORDER BY embedding <-> %s::vector
                LIMIT %s;
                """,
                (vec_lit, top_k),
            )
            rows = cur.fetchall()

    return rows


def answer_question(question: str) -> str:
    """RAG: retrieve top chunks and answer with Gemini."""
    _, qa_model = init_vertex()
    chunks = search_similar_chunks(question, top_k=5)

    if not chunks:
        return "I couldn't find any content in the database yet. Try uploading a document first."

    context_blocks = []
    for _, source_uri, chunk_index, content in chunks:
        context_blocks.append(f"[{source_uri} – chunk {chunk_index}]\n{content}")

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a helpful assistant. Use ONLY the context below to answer the user's question.
If the answer is not clearly supported by the context, say you are not sure.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER (clear and concise):
    """.strip()

    resp = qa_model.generate_content(prompt)
    return resp.text.strip()


# ---------------- FastAPI app ----------------

app = FastAPI(title="RAG Demo (TXT + PDF)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # relax for now – can tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    # Make sure DB + models are ready
    init_db_schema()
    init_vertex()


@app.get("/", response_class=HTMLResponse)
def home():
    # Very simple UI – we’ll improve later, for now just TXT/PDF upload + ask
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Demo</title>
        <style>
            body { font-family: system-ui, -apple-system, Arial, sans-serif; margin: 2rem; background: #0f172a; color: #e5e7eb; }
            h1 { color: #facc15; }
            .card { background: #020617; padding: 1.5rem; border-radius: 0.75rem; margin-bottom: 1.5rem; border: 1px solid #1f2937; }
            label { display: block; margin-bottom: 0.5rem; }
            input[type="file"], input[type="text"], textarea {
                width: 100%; padding: 0.5rem; border-radius: 0.5rem;
                border: 1px solid #374151; background: #020617; color: #e5e7eb;
            }
            button {
                margin-top: 0.75rem; padding: 0.6rem 1.2rem; border-radius: 999px;
                border: none; background: #4f46e5; color: white; font-weight: 600;
                cursor: pointer;
            }
            button:hover { background: #6366f1; }
            .response-box { white-space: pre-wrap; margin-top: 0.75rem; padding: 0.75rem; background: #020617; border-radius: 0.5rem; border: 1px solid #1f2937; }
        </style>
    </head>
    <body>
        <h1>RAG Demo (TXT + PDF)</h1>

        <div class="card">
            <h2>1. Upload a document</h2>
            <p>Supported: <strong>.txt</strong> and <strong>.pdf</strong>.</p>
            <form id="upload-form" enctype="multipart/form-data">
                <label for="file">Choose file:</label>
                <input type="file" id="file" name="file" required />
                <button type="submit">Upload & Embed</button>
            </form>
            <div id="upload-result" class="response-box"></div>
        </div>

        <div class="card">
            <h2>2. Ask a question</h2>
            <form id="ask-form">
                <label for="question">Your question:</label>
                <textarea id="question" name="question" rows="3" required></textarea>
                <button type="submit">Ask</button>
            </form>
            <div id="answer" class="response-box"></div>
        </div>

        <script>
            const uploadForm = document.getElementById("upload-form");
            const uploadResult = document.getElementById("upload-result");
            const askForm = document.getElementById("ask-form");
            const answerBox = document.getElementById("answer");

            uploadForm.addEventListener("submit", async (e) => {
                e.preventDefault();
                uploadResult.textContent = "Uploading and embedding…";
                const formData = new FormData(uploadForm);
                try {
                    const resp = await fetch("/upload", {
                        method: "POST",
                        body: formData
                    });
                    const data = await resp.json();
                    uploadResult.textContent = data.message || JSON.stringify(data);
                } catch (err) {
                    uploadResult.textContent = "Error: " + err;
                }
            });

            askForm.addEventListener("submit", async (e) => {
                e.preventDefault();
                answerBox.textContent = "Thinking…";
                const question = document.getElementById("question").value;
                try {
                    const resp = await fetch("/ask", {
                        method: "POST",
                        headers: { "Content-Type": "application/x-www-form-urlencoded" },
                        body: new URLSearchParams({ question })
                    });
                    const data = await resp.json();
                    answerBox.textContent = data.answer || JSON.stringify(data);
                } catch (err) {
                    answerBox.textContent = "Error: " + err;
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        filename = file.filename or "uploaded_file"
        raw = await file.read()

        if not raw:
            return JSONResponse(
                {"message": "Uploaded file is empty."}, status_code=400
            )

        # Detect PDF vs text
        is_pdf = (
            file.content_type == "application/pdf"
            or filename.lower().endswith(".pdf")
        )

        if is_pdf:
            text = extract_text_from_pdf(raw)
        else:
            # Assume UTF-8 text for non-PDF
            text = raw.decode("utf-8", errors="ignore")

        if not text.strip():
            return JSONResponse(
                {"message": "No extractable text found in the file."}, status_code=400
            )

        num_chunks = store_document(source_uri=filename, full_text=text)

        return JSONResponse(
            {
                "message": f"Uploaded '{filename}' and stored {num_chunks} chunks.",
                "filename": filename,
                "chunks": num_chunks,
            }
        )

    except Exception as e:
        return JSONResponse(
            {"message": f"Error during upload: {e!r}"}, status_code=500
        )


@app.post("/ask")
async def ask(question: str = Form(...)):
    try:
        answer = answer_question(question)
        return JSONResponse({"answer": answer})
    except Exception as e:
        return JSONResponse(
            {"answer": f"Error while answering question: {e!r}"}, status_code=500
        )
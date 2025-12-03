import os
import io
from typing import List

import psycopg2
from google.cloud import storage
import vertexai
from vertexai.language_models import TextEmbeddingModel
from PyPDF2 import PdfReader

# ===== Config =====
# We try to load these from environment variables, but defaults are provided based on your .env
PROJECT_ID = os.getenv("GCP_PROJECT", "starry-journal-480011-m8")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")  # <--- WE WILL SET THIS IN TERMINAL

# DB Config (Matches your existing ingest_pg.py settings)
DB_HOST = os.getenv("DB_HOST", "35.184.43.16")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "ragdb")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Bhatti@512") 

EMBEDDING_MODEL_NAME = "text-embedding-004"

# ===== Helpers (Reused from your existing project) =====

def get_connection():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
    )

def init_vertex():
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    return TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)

def get_embedding(model, text):
    return model.get_embeddings([text])[0].values

def chunk_text(text: str, max_chars: int = 800, overlap: int = 200) -> List[str]:
    """Splits text into smaller pieces so the AI can digest it."""
    text = text.replace("\r\n", "\n")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks

def read_pdf_from_bytes(file_bytes) -> str:
    """Helper to extract text from a PDF file downloaded from Cloud Storage."""
    reader = PdfReader(io.BytesIO(file_bytes))
    text = []
    for page in reader.pages:
        txt = page.extract_text()
        if txt:
            text.append(txt)
    return "\n".join(text)

# ===== Main Logic =====

def ingest_bucket(bucket_name: str):
    if not bucket_name:
        print("❌ Error: Bucket name is missing.")
        return

    print(f"🔌 Connecting to GCS Bucket: {bucket_name}")
    
    # 1. Initialize Google Cloud clients
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    embed_model = init_vertex()
    
    # 2. List all files in the bucket
    blobs = list(bucket.list_blobs())
    print(f"📂 Found {len(blobs)} files in bucket.")

    conn = get_connection()
    try:
        for blob in blobs:
            # We only want .txt or .pdf files
            if not (blob.name.endswith(".txt") or blob.name.endswith(".pdf")):
                print(f"⏭️  Skipping {blob.name} (file type not supported)")
                continue

            print(f"📥 Processing: {blob.name} ...")
            
            # 3. Download the file content into memory
            file_bytes = blob.download_as_bytes()
            
            # 4. Extract text depending on file type
            text_content = ""
            try:
                if blob.name.endswith(".pdf"):
                    text_content = read_pdf_from_bytes(file_bytes)
                else:
                    text_content = file_bytes.decode("utf-8", errors="ignore")
            except Exception as e:
                print(f"   ❌ Error reading file: {e}")
                continue

            if not text_content.strip():
                print("   ⚠️ Empty file, skipping.")
                continue

            # 5. Chunk the text
            chunks = chunk_text(text_content)
            print(f"   → Generated {len(chunks)} chunks.")

            # 6. Save to Database (Embed & Insert)
            with conn.cursor() as cur:
                # Use "gs://" format so we know it came from the cloud
                source_uri = f"gs://{bucket_name}/{blob.name}"
                
                # Delete old chunks for this specific file (to avoid duplicates)
                cur.execute("DELETE FROM documents WHERE source_uri = %s;", (source_uri,))
                
                # Insert new chunks
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
            print("   ✅ Saved to DB.")

    finally:
        conn.close()

if __name__ == "__main__":
    ingest_bucket(BUCKET_NAME)
import os
import shutil
import json
import re
import requests
import chromadb
import trafilatura
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

# -----------------------------
# NLTK SETUP (SAFE)
# -----------------------------
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

# -----------------------------
# PATH SETUP (PROJECT ROOT SAFE)
# -----------------------------
CURRENT_FILE = os.path.abspath(__file__)                 # project_bot/database/vectordb/vectordb.py
VECTORDIR = os.path.dirname(CURRENT_FILE)                # project_bot/database/vectordb
DATABASE_DIR = os.path.dirname(VECTORDIR)                # project_bot/database
PROJECT_ROOT = os.path.dirname(DATABASE_DIR)             # project_bot

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHUNKS_DIR = os.path.join(DATA_DIR, "chunks")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")

FAQ_PATH = os.path.join(EMBEDDINGS_DIR, "faq_embeddings.json")
VECTOR_DB_PATH = os.path.join(PROJECT_ROOT, "vector_db")

os.makedirs(CHUNKS_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# -----------------------------
# STRONG IDEMPOTENCY (VECTOR DB)
# -----------------------------
print("🔄 Checking vector DB state...")

if os.path.exists(VECTOR_DB_PATH):
    print("⚠ Existing vector_db found. Deleting for fresh rebuild...")
    shutil.rmtree(VECTOR_DB_PATH)

os.makedirs(VECTOR_DB_PATH, exist_ok=True)
print("✅ Fresh vector_db directory ready")

# -----------------------------
# TARGET WEB PAGES
# -----------------------------
TARGET_PAGES = [
    "https://tkrcet.ac.in/",
    "https://tkrcet.ac.in/vision-mission/",
    "https://tkrcet.ac.in/about-the-campus/",
    "https://tkrcet.ac.in/chairmans-message/",
    "https://tkrcet.ac.in/secretarys-message/",
    "https://tkrcet.ac.in/treasurers-message/",
    "https://tkrcet.ac.in/principal/",
    "https://tkrcet.ac.in/vice-principal/",
    "https://tkrcet.ac.in/admission-procedure/",
    "https://tkrcet.ac.in/fee-structure/",
    "https://tkrcet.ac.in/computer-science-and-engineering/",
    "https://tkrcet.ac.in/cse-ds-csd/",
]

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

# -----------------------------
# LOAD FAQ DATA
# -----------------------------
with open(FAQ_PATH, encoding="utf-8") as f:
    faqs = json.load(f)

# -----------------------------
# EMBEDDING MODEL
# -----------------------------
embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")

# -----------------------------
# TEXT CLEANING
# -----------------------------
def clean_text(html: str) -> str:
    extracted = trafilatura.extract(html, include_comments=False)
    if not extracted:
        return ""

    text = re.sub(r"\s+", " ", extracted).strip()
    sentences = [s for s in sent_tokenize(text) if len(s.split()) > 5]
    return " ".join(sentences)

# -----------------------------
# CHUNKING
# -----------------------------
def chunk_text(text: str):
    words = text.split()
    chunks = []
    i = 0

    while i < len(words):
        chunk = words[i:i + CHUNK_SIZE]
        chunks.append(" ".join(chunk))
        i += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks

# -----------------------------
# SCRAPING
# -----------------------------
def scrape_page(url: str) -> str:
    print(f"🌐 Scraping → {url}")
    try:
        res = requests.get(url, timeout=20)
        res.raise_for_status()
        return clean_text(res.text)
    except Exception as e:
        print(f"❌ Failed to scrape {url}: {e}")
        return ""

# -----------------------------
# CHROMA CLIENT
# -----------------------------
client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

for col in ["faqs_collection", "webdocs_collection"]:
    try:
        client.delete_collection(col)
    except Exception:
        pass

faqs_collection = client.get_or_create_collection(
    name="faqs_collection",
    metadata={"hnsw:space": "cosine"}
)

web_collection = client.get_or_create_collection(
    name="webdocs_collection",
    metadata={"hnsw:space": "cosine"}
)

# -----------------------------
# STORE FAQ VECTORS
# -----------------------------
print("\n🚀 Storing FAQ vectors...")

faqs_collection.add(
    ids=[f"faq_{f['faq_id']:05d}" for f in faqs],
    documents=[f["combined_text"] for f in faqs],
    embeddings=[f["embedding"] for f in faqs],
    metadatas=[
        {
            "type": "faq",
            "faq_id": f["faq_id"],
            "question": f["question"],
            "answer": f["answer"]
        }
        for f in faqs
    ]
)

print(f"✅ FAQ stored: {faqs_collection.count()}")

# -----------------------------
# BUILD WEB VECTORS + SAVE FILES
# -----------------------------
print("\n🚀 Building web document vectors...")

all_chunks = []
all_embeddings = []

ids, documents, embeddings, metadatas = [], [], [], []
counter = 0

for url in TARGET_PAGES:
    text = scrape_page(url)

    if not text or len(text) < 200:
        continue

    chunks = chunk_text(text)

    for idx, chunk in enumerate(chunks):
        chunk_id = f"{abs(hash(url))}_{idx}"
        vector = embedder.encode(chunk, normalize_embeddings=True).tolist()

        ids.append(chunk_id)
        documents.append(chunk)
        embeddings.append(vector)
        metadatas.append({"source_url": url})

        all_chunks.append({
            "id": chunk_id,
            "source_url": url,
            "text": chunk
        })

        all_embeddings.append({
            "id": chunk_id,
            "embedding": vector
        })

        counter += 1

# -----------------------------
# SAVE CHUNKS & EMBEDDINGS
# -----------------------------
with open(os.path.join(CHUNKS_DIR, "web_chunks.json"), "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

with open(os.path.join(EMBEDDINGS_DIR, "web_embeddings.json"), "w", encoding="utf-8") as f:
    json.dump(all_embeddings, f, indent=2)

# -----------------------------
# STORE IN VECTOR DB
# -----------------------------
web_collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)

# -----------------------------
# FINAL REPORT
# -----------------------------
print("\n🎯 FINAL RESULT")
print("FAQ Count:", faqs_collection.count())
print("Web Count:", web_collection.count())
print("📁 Chunks saved to:", CHUNKS_DIR)
print("📁 Embeddings saved to:", EMBEDDINGS_DIR)
print("🔥 Vector database rebuilt successfully")

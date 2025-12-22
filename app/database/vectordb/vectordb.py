import json
import chromadb

# ---------- Load JSON files ----------
with open("faq_embeddings.json", encoding="utf-8") as f:
    faqs = json.load(f)

with open("unified_vectors.json", encoding="utf-8") as f:
    webdocs = json.load(f)

# ---------- Create Chroma client ----------
client = chromadb.Client()

faqs_collection = client.create_collection(
    name="faqs_collection"
)

webdocs_collection = client.create_collection(
    name="webdocs_collection"
)

# ---------- Store FAQs (FIXED) ----------
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

print("✅ FAQs stored:", faqs_collection.count())

# ---------- Store Web-scraped chunks ----------
webdocs_collection.add(
    ids=[r["chunk_id"] for r in webdocs],
    documents=[r["text"] for r in webdocs],
    embeddings=[r["embedding"] for r in webdocs],
    metadatas=[r["metadata"] for r in webdocs]
)

print("✅ Web docs stored:", webdocs_collection.count())

# ---------- Final verification ----------
print("FAQs count:", faqs_collection.count())
print("Web docs count:", webdocs_collection.count())

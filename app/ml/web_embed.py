import json
from sentence_transformers import SentenceTransformer

INPUT_FILE = "chunks.json"
OUTPUT_FILE = "chunks_embeddings.json"

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [c["text"] for c in chunks]

embeddings = model.encode(
    texts,
    batch_size=16,
    normalize_embeddings=True,
    show_progress_bar=True
)

for i, emb in enumerate(embeddings):
    chunks[i]["embedding"] = emb.tolist()

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

print("Embeddings generated and saved")

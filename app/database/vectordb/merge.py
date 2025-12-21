import json

# Load files safely
with open("chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

with open("chunks_embeddings.json", encoding="utf-8") as f:
    embeddings = json.load(f)

# Safety check
assert len(chunks) == len(embeddings), "Length mismatch!"

merged = []

for i in range(len(chunks)):
    merged.append({
        "chunk_id": f"chunk_{i:05d}",
        "text": chunks[i]["text"],          # ALWAYS from chunks.json
        "embedding": embeddings[i]["embedding"],
        "metadata": {
            "section": chunks[i]["section"],
            "source_url": chunks[i]["source_url"]
        }
    })

# Save unified JSON
with open("unified_vectors.json", "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

print(f"✅ Merged {len(merged)} records successfully")

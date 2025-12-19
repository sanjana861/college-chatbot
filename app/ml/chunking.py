import json

INPUT_FILE = "webdata_clean.json"
OUTPUT_FILE = "chunks.json"

CHUNK_SIZE = 500
OVERLAP = 100

def chunk_text(text):
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - OVERLAP
    return chunks

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    site_data = json.load(f)

chunks = []

for page in site_data.values():
    for section, content in page["sections"].items():
        for text in content["texts"]:
            for c in chunk_text(text):
                chunks.append({
                    "text": c,
                    "section": section,
                    "source_url": page["url"]
                })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)

print(f"{len(chunks)} chunks saved to chunks.json")

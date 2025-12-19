import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ---------------------------
# 1. Load the model
# ---------------------------
print("🔹 Loading BGE-Large model...")
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# ---------------------------
# 2. Load FAQ JSON file
# ---------------------------
file_path = r"C:\Users\chand\OneDrive\Desktop\embeddings\faq_rows.json"

print("🔹 Loading FAQ JSON...")
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert JSON → DataFrame
df = pd.DataFrame(data)

# Detect columns
question_col = df.columns[0]
answer_col   = df.columns[1]

print("Detected columns:", question_col, answer_col)

# Drop empty rows
df = df.dropna(subset=[question_col, answer_col])

# Combine Question + Answer
faqs = (
    df[question_col].astype(str)
    + " | ANSWER: "
    + df[answer_col].astype(str)
).tolist()

print(f"Total FAQs processed: {len(faqs)}")

# ---------------------------
# 3. Generate embeddings
# ---------------------------
print("🔹 Generating embeddings...")
embeddings = model.encode(
    faqs,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("Embedding shape:", embeddings.shape)

# ---------------------------
# 4. Save embeddings (.npy)
# ---------------------------
np.save("faq_embeddings.npy", embeddings)

# ---------------------------
# 5. Save text data
# ---------------------------
with open("faq_texts.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "questions": df[question_col].tolist(),
            "answers": df[answer_col].tolist(),
            "combined_text": faqs
        },
        f,
        indent=4,
        ensure_ascii=False
    )

# ---------------------------
# 6. Save embeddings + metadata (JSON)
# ---------------------------
faq_records = []

for idx, vector in enumerate(embeddings):
    faq_records.append({
        "faq_id": idx,
        "question": df.iloc[idx][question_col],
        "answer": df.iloc[idx][answer_col],
        "combined_text": faqs[idx],
        "embedding": vector.tolist()
    })

with open("new_faq_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(faq_records, f, indent=2, ensure_ascii=False)

print("\n✅ Generated files:")
print(" - faq_embeddings.npy")
print(" - faq_texts.json")
print(" - new_faq_embeddings.json")

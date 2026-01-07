import chromadb
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------ MODEL LOAD ------------------

# BGE Embedding model
embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Faster but still strong reranker
reranker = CrossEncoder("BAAI/bge-reranker-base")

# ✅ PURE LLM (decoder-only, FREE, FAST)
LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
generator = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# ------------------ CHROMA ------------------
client = chromadb.PersistentClient(path="vector_db")
faqs_collection = client.get_collection("faqs_collection")
webdocs_collection = client.get_collection("webdocs_collection")

sample = webdocs_collection.get(ids=webdocs_collection.peek()["ids"])

for i, d in enumerate(sample["documents"]):
    print(f"\n---- DOC {i} ----")
    print(d[:400])
    print(sample["metadatas"][i])

# ------------------ RETRIEVAL ------------------
def retrieve(collection, query, top_k=30):
    query_emb = embedder.encode(query, normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["documents", "metadatas"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    ids = results["ids"][0]

    seen = set()
    unique_docs, unique_metas, unique_ids = [], [], []

    for d, m, i in zip(docs, metas, ids):
        if i not in seen and d.strip():
            seen.add(i)
            unique_docs.append(d)
            unique_metas.append(m)
            unique_ids.append(i)

    return unique_docs, unique_metas, unique_ids

# ------------------ RERANK ------------------
def rerank(query, docs, metas, ids, top_k=5):
    pairs = [[query, d] for d in docs]
    scores = reranker.predict(pairs)
    scores = np.array(scores)

    sorted_idx = np.argsort(-scores)

    used_texts = set()
    reranked = []

    for i in sorted_idx:
        text = docs[i].strip()
        if text not in used_texts and text:
            reranked.append({
                "score": float(scores[i]),
                "document": text,
                "metadata": metas[i],
                "id": ids[i]
            })
            used_texts.add(text)

        if len(reranked) == top_k:
            break

    return reranked

# ------------------ AUGMENTATION ------------------
def build_context(reranked_results, max_chars=3500):
    blocks = []
    total_chars = 0

    for r in reranked_results:
        text = r["document"]
        meta = r["metadata"]
        block = f"Source: {meta.get('source','web')}\nContent:\n{text}\n\n"

        if total_chars + len(block) > max_chars:
            break

        blocks.append(block)
        total_chars += len(block)

    return "\n".join(blocks)

# ------------------ PROMPT ------------------
def build_prompt(query, context):
    return f"""
You are a highly accurate college assistant chatbot.

RULES:
- Answer ONLY using the provided context
- If the answer is NOT in context, reply exactly: "I do not know."
- Do NOT guess
- Be concise and clear
- If multiple relevant points exist, summarize cleanly

CONTEXT:
{context}

QUESTION:
{query}

FINAL ANSWER:
"""

# ------------------ GENERATION ------------------
def generate_llama_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(generator.device)

    output = generator.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,                 # 🔑 stable + faster
        temperature=0.5,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # 🔑 remove prompt echo (pure LLM behavior)
    return decoded[len(prompt):].strip()

# ------------------ FULL PIPELINE ------------------
def rag_answer(query):
    faq_docs, faq_metas, faq_ids = retrieve(faqs_collection, query)
    web_docs, web_metas, web_ids = retrieve(webdocs_collection, query)

    all_docs = faq_docs + web_docs
    all_metas = faq_metas + web_metas
    all_ids = faq_ids + web_ids

    reranked = rerank(query, all_docs, all_metas, all_ids, top_k=5)

    if len(reranked) == 0:
        return "I do not know.", []

    context = build_context(reranked)
    prompt = build_prompt(query, context)
    answer = generate_llama_answer(prompt)

    # ❗ RETURN STRUCTURE UNCHANGED
    return {
        "status": "success",
        "reply": answer
    }
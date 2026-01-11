import chromadb
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------ MODEL LOAD ------------------

# BGE Embedding model
embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Faster but still strong reranker
reranker = CrossEncoder("BAAI/bge-reranker-base")

# Mistral 7B Instruct
#LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
LLM_MODEL = "google/flan-t5-large"
#LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


"""tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
generator = AutoModelForCausalLM.from_pretrained(
   LLM_MODEL,
   torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
   device_map="auto"
)"""
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
generator = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)


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
        include=["documents", "metadatas"]  # <-- remove "ids"
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    ids = results["ids"][0]   # <-- still available automatically

    # ---- DEDUPLICATE ----
    seen = set()
    unique_docs = []
    unique_metas = []
    unique_ids = []

    for d, m, i in zip(docs, metas, ids):
        if i not in seen and d.strip() != "":
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
        if text not in used_texts and text != "":
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
def build_context(reranked_results):
    blocks = []
    for r in reranked_results:
        text = r["document"]
        meta = r["metadata"]
        block = f"Source: {meta.get('source','web')}\nContent:\n{text}"
        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


# ------------------ PROMPT ------------------
def build_prompt(query, context):
    prompt = f"""
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
    return prompt


# ------------------ GENERATION ------------------
def generate_llama_answer(prompt):

    inputs = tokenizer(prompt, return_tensors="pt").to(generator.device)

    output = generator.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.9,
        top_p=0.95,
        do_sample=False
    )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    return answer
    




# ------------------ FULL PIPELINE ------------------
def rag_answer(query):
    # -------- Retrieve from FAQ --------
    faq_docs, faq_metas, faq_ids = retrieve(faqs_collection, query)

    # -------- Retrieve from Web --------
    web_docs, web_metas, web_ids = retrieve(webdocs_collection, query)

    # -------- Merge Results --------
    all_docs = faq_docs + web_docs
    all_metas = faq_metas + web_metas
    all_ids = faq_ids + web_ids

    # -------- Rerank Combined --------
    reranked = rerank(query, all_docs, all_metas, all_ids, top_k=5)

    if len(reranked) == 0:
        return "I do not know.", []

    # -------- Build Context --------
    context = build_context(reranked)

    # -------- Prompt --------
    prompt = build_prompt(query, context)

    # -------- Generate --------
    answer = generate_llama_answer(prompt)

    return answer, reranked



# ------------------ TEST ------------------
if __name__ == "__main__":
    query = "What is the admission procedure?"
    
    answer, evidence = rag_answer(query)


    print("\n================= ANSWER =================\n")
    print(answer)

    print("\n=========== SUPPORTING CONTEXT ===========\n")
    for r in evidence:
        print(f"SCORE: {r['score']}")
        print(r["document"][:250], "...")
        print("------------------------------------------")

print("FAQ count:", faqs_collection.count())
print("Web count:", webdocs_collection.count())

res = webdocs_collection.peek()
print(len(res["ids"]))
print(len(set(res["ids"])))
  
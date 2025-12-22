
import json
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import numpy as np

class RAGSystem:
    def __init__(self, chunks_path='app/database/vectordb/unified_vectors.json', embedding_model='all-MiniLM-L6-v2', cross_encoder_model='cross-encoder/ms-marco-MiniLM-L-6-v2', llm_model='google/flan-t5-base'):
        # Load data
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        # Initialize models
        self.embedding_model = SentenceTransformer(embedding_model)
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        self.tokenizer = T5Tokenizer.from_pretrained(llm_model)
        self.llm = T5ForConditionalGeneration.from_pretrained(llm_model)
        
        # Re-generate embeddings in memory
        self.embeddings = self.embedding_model.encode([chunk['text'] for chunk in self.chunks], convert_to_tensor=True)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings.cpu().detach().numpy())

    def search(self, query, k=5):
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        query_embedding = query_embedding.cpu().detach().numpy()
        distances, indices = self.index.search(np.array([query_embedding]), k)
        return [self.chunks[i] for i in indices[0]]

    def rerank(self, query, documents):
        scores = self.cross_encoder.predict([(query, doc['text']) for doc in documents])
        ranked_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return [doc for score, doc in ranked_docs]

    def generate(self, query, documents, temperature=0.7, top_k=50, top_p=0.9):
        context = " ".join([doc['text'] for doc in documents])
        input_text = f"question: {query} context: {context}"
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids

        outputs = self.llm.generate(
            input_ids,
            max_length=200,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=1,
            do_sample=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def __call__(self, query):
        retrieved_docs = self.search(query)
        reranked_docs = self.rerank(query, retrieved_docs)
        answer = self.generate(query, reranked_docs)
        return answer

if __name__ == '__main__':
    rag_system = RAGSystem()
    query = "What is the admission process?"
    answer = rag_system(query)
    print(f"Query: {query}")
    print(f"Answer: {answer}")

import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

class CollegeBotLLM:
    def __init__(self):
        # 1. DOWNLOAD MODEL (Small & Fast: Phi-3-Mini GGUF)
        # This only downloads once. It's ~2.3GB - perfect for a prototype.
        model_path = hf_hub_download(
            repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
            filename="Phi-3-mini-4k-instruct-q4.gguf"
        )
        
        # 2. INITIALIZE LLM
        # n_ctx is the context window (input + output length)
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048, 
            n_threads=4, # Adjust based on your CPU cores
            verbose=False
        )

    def generate_response(self, user_query, retrieved_context):
        """
        Takes the query and retrieved data to generate a final answer.
        """
        # 3. PROMPT ENGINEERING (Crucial for RAG)
        prompt = f"""<|system|>
You are a helpful assistant for college-related queries. 
Use ONLY the following context to answer the question. 
If the answer isn't in the context, say "I don't have that information in my database."

Context:
{retrieved_context}
<|end|>
<|user|>
{user_query}<|end|>
<|assistant|>"""

        # 4. GENERATION (Decoding and Logic)
        response = self.llm(
            prompt,
            max_tokens=256,
            temperature=0.2, # Low temp = more factual/less creative
            stop=["<|end|>", "\n\n"]
        )

        return response['choices'][0]['text'].strip()

# --- HOW TO USE IN YOUR PROJECT ---
if __name__ == "__main__":
    bot = CollegeBotLLM()
    
    # Simulate data coming from your "Completed" RAG steps
    sample_context = "The Computer Science HOD is Dr. Smith. Office hours are 10 AM to 12 PM."
    sample_query = "Who is the HOD of CS and when can I meet him?"
    
    answer = bot.generate_response(sample_query, sample_context)
    print(f"Bot Response: {answer}")
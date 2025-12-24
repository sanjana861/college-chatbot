import ollama
import chromadb

# --- Configuration ---
CHROMA_PATH = "chroma_db"  # Path to your existing database
COLLECTION_NAME = "college_data"

def get_context_from_db(query):
    """
    Connects to your existing ChromaDB and retrieves relevant documents.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    
    # Retrieve top 5 matches
    results = collection.query(
        query_texts=[query],
        n_results=5
    )
    
    # Combine retrieved documents into a single string
    context = "\n".join(results['documents'][0])
    return context

def generate_college_response(user_query, user_role="student"):
    """
    Combines retrieval and LLM generation.
    """
    # 1. Retrieve context
    context = get_context_from_db(user_query)

    # 2. Define System Instructions
    system_instructions = """
    You are an AI-powered College Assistant Chatbot.
    RULES:
    1. Only answer college-related questions.
    2. If the query is unrelated, say: "I can assist only with college-related queries."
    3. Use ONLY the retrieved context provided. Do NOT guess.
    4. If information is missing, say: "This information is currently not available. Please contact the college admin."
    5. Mention portals (ERP, Library, etc.) when relevant.
    """

    # 3. Construct the User Prompt
    prompt = f"""
    Context:
    {context}

    User Type: {user_role}
    User Query: "{user_query}"

    Answer:
    """

    # 4. Get response from Ollama
    response = ollama.generate(
        model='llama3.2:1b',
        system=system_instructions,
        prompt=prompt,
        options={
            'temperature': 0.3,
            'num_ctx': 4096
        }
    )
    return response['response']

# --- Execution ---
if __name__ == "__main__":
    query = "How do I access the ERP portal for exam results?"
    print(f"Query: {query}")
    print("-" * 30)
    print(f"Chatbot: {generate_college_response(query)}")
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid

from backend.intent_router import detect_intent
from backend.state_manager import set_state, clear_state
from database.sql.sql_retrevial import handle_user_query as sql_handler
from services.rag import rag_answer


app = FastAPI(title="College Hybrid Backend")

# -------- CORS (ONE TIME FIX) --------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Request Model --------
class QueryRequest(BaseModel):
    message: str
    session_id: str | None = None


# -------- Health Check --------
@app.get("/")
def health():
    return {"status": "Backend running successfully"}


# -------- Main Chat Endpoint --------
@app.post("/query")
def query_router(req: QueryRequest):
    session_id = req.session_id or str(uuid.uuid4())
    user_query = req.message

    intent = detect_intent(user_query)

    # -------- SQL --------
    """
    if intent == "sql":
        result = sql_handler(user_query)

        if result.get("status") == "need_more_info":
            set_state(session_id, {"intent": "sql"})
            return {
                "answer": result["message"],
                "session_id": session_id
            }

        clear_state(session_id)
        return {
            "answer": str(result),
            "session_id": session_id
        }
    """
    if intent == "sql":
        result = sql_handler(user_query)

        if result.get("status") == "need_more_info":
            return {
                "answer": result["message"],
                "session_id": session_id
            }

        return {
            "answer": result.get("answer", "No result found."),
            "session_id": session_id
        }
    
    # -------- RAG --------
    answer, _ = rag_answer(user_query)

    return {
        "answer": answer,
        "session_id": session_id
    }

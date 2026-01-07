from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uuid

from backend.intent_router import detect_intent
from backend.state_manager import set_state, clear_state
from database.sql.sql_retrevial import handle_user_query
from services.rag import rag_answer

app = FastAPI(title="College Hybrid Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str
    session_id: str | None = None

"""@app.post("/query")
def query_router(req: Query):
    session_id = req.session_id or str(uuid.uuid4())
    intent = detect_intent(req.query)

    if intent == "sql":
        result = handle_user_query(req.query)
        if result["status"] == "need_more_info":
            set_state(session_id, {"intent": "sql"})
            result["session_id"] = session_id
            return result
        clear_state(session_id)
        return result

    return rag_answer(req.query)"""

@app.post("/query")
def query_router(req: Query):
    session_id = req.session_id or str(uuid.uuid4())
    intent = detect_intent(req.query)

    if intent == "sql":
        result = handle_user_query(req.query)

        if result["status"] == "need_more_info":
            return {
                "status": "need_more_info",
                "reply": result["message"],
                "session_id": session_id
            }

        rows = result.get("data", [])

        if not rows:
            return {
                "status": "success",
                "reply": "No matching students found."
            }

        # format SQL rows into readable text
        lines = []
        for r in rows:
            lines.append(
                f"Roll No: {r[0]}, Name: {r[1]}, Branch: {r[2]}, CGPA: {r[3]}"
            )

        return {
            "status": "success",
            "reply": "\n".join(lines)
        }

    # RAG path
    return rag_answer(req.query)


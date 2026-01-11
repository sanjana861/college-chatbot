import json
import os
from typing import List, Dict

from services.rag import rag_answer

# ---------------- PATH SETUP (FIXED) ----------------
CURRENT_FILE = os.path.abspath(__file__)          # backend/intent_router.py
BACKEND_DIR = os.path.dirname(CURRENT_FILE)       # backend/
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)       # project_bot/

PORTAL_REGISTRY_PATH = os.path.join(
    PROJECT_ROOT,
    "portal_registry.json"
)

# ---------------- LOAD PORTAL REGISTRY ----------------
with open(PORTAL_REGISTRY_PATH, "r", encoding="utf-8") as f:
    PORTALS = json.load(f)

# ---------------- SQL KEYWORDS ----------------
SQL_KEYWORDS = [
    "name", "cgpa", "branch", "placed",
    "roll", "female", "male"
]

# ---------------- INTENT DETECTOR ----------------
def detect_intent(query: str) -> str:
    """
    Priority:
    1. SQL
    2. Navigation
    3. RAG
    """
    q = query.lower()

    for k in SQL_KEYWORDS:
        if k in q:
            return "sql"

    for portal in PORTALS.values():
        for kw in portal["keywords"]:
            if kw in q:
                return "navigation"

    return "rag"


# ---------------- PORTAL MATCHER ----------------
def match_portals(query: str) -> List[Dict]:
    q = query.lower()
    matched = []

    for portal in PORTALS.values():
        for kw in portal["keywords"]:
            if kw in q:
                matched.append({
                    "label": portal["label"],
                    "url": portal["url"],
                    "capabilities": portal["capabilities"]
                })
                break

    return matched

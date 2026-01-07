SQL_KEYWORDS = [
    "student","cgpa","branch","placed","roll","female","male"
]

def detect_intent(query):
    q = query.lower()
    for k in SQL_KEYWORDS:
        if k in q:
            return "sql"
    return "rag"

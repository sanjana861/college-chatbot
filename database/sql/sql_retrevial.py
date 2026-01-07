import sqlite3
import re
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "students.db")

def extract_placement(q):
    q = q.lower()
    if re.search(r"\bnot\s+placed\b|\bunplaced\b", q):
        return False
    if re.search(r"\bplaced\b", q):
        return True
    return None

def handle_user_query(query):
    filters = {}

    if "female" in query.lower(): filters["gender"] = "female"
    if "male" in query.lower(): filters["gender"] = "male"

    if "cse" in query.lower(): filters["branch"] = "CSE"
    if "ece" in query.lower(): filters["branch"] = "ECE"

    placed = extract_placement(query)
    if placed is not None:
        filters["placed"] = placed

    match = re.search(r"(above|below)\s+(\d+(\.\d+)?)", query.lower())
    if match:
        op = ">" if match.group(1) == "above" else "<"
        filters["cgpa"] = (op, float(match.group(2)))

    if not filters:
        return {
            "status": "need_more_info",
            "message": "Please specify student attributes."
        }

    sql = "SELECT roll_no,name,branch,cgpa FROM students WHERE 1=1"
    params = []

    if "gender" in filters:
        sql += " AND LOWER(gender)=?"
        params.append(filters["gender"])

    if "branch" in filters:
        sql += " AND branch=?"
        params.append(filters["branch"])

    if "cgpa" in filters:
        op, v = filters["cgpa"]
        sql += f" AND cgpa {op} ?"
        params.append(v)

    if "placed" in filters:
        sql += " AND company_placed IS NOT NULL" if filters["placed"] else " AND company_placed IS NULL"

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()

    return {
        "status": "success",
        "count": len(rows),
        "data": rows
    }

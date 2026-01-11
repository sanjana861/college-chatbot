import sqlite3
import re
import os

# ---------------- PATH ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DB_PATH = os.path.join(PROJECT_ROOT, "students.db")


# ---------------- CGPA PARSER ----------------
def extract_cgpa(query: str):
    """
    Supports:
    - above 8
    - greater than 9
    - more than 7.5
    - below 8
    - less than 6
    """
    q = query.lower()

    patterns = [
        (r"(above|greater than|more than)\s+(\d+(\.\d+)?)", ">"),
        (r"(below|less than)\s+(\d+(\.\d+)?)", "<")
    ]

    for pattern, op in patterns:
        match = re.search(pattern, q)
        if match:
            return op, float(match.group(2))

    return None


# ---------------- MAIN HANDLER ----------------
def handle_user_query(query):
    filters = {}
    q = query.lower()

    # -------- Gender --------
    if "female" in q:
        filters["gender"] = "female"
    elif "male" in q:
        filters["gender"] = "male"

    # -------- Branch --------
    if "cse" in q:
        filters["branch"] = "CSE"
    elif "ece" in q:
        filters["branch"] = "ECE"
    elif "csd" in q:
        filters["branch"] = "CSD"

    # -------- CGPA --------
    cgpa_filter = extract_cgpa(q)
    if cgpa_filter:
        filters["cgpa"] = cgpa_filter

    # ❌ If still no filters → ask user
    if not filters:
        return {
            "status": "need_more_info",
            "message": "Please specify student attributes like CGPA, branch, or gender."
        }

    # -------- SQL BUILD --------
    sql = "SELECT name, roll_no, branch, cgpa FROM students WHERE 1=1"
    params = []

    if "gender" in filters:
        sql += " AND LOWER(gender)=?"
        params.append(filters["gender"])

    if "branch" in filters:
        sql += " AND branch=?"
        params.append(filters["branch"])

    if "cgpa" in filters:
        op, value = filters["cgpa"]
        sql += f" AND cgpa {op} ?"
        params.append(value)

    # -------- EXECUTE --------
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return {
            "status": "success",
            "answer": "No students found matching the criteria."
        }

    # -------- FORMAT OUTPUT --------
    result_lines = [
        f"{name} ({roll}) - {branch}, CGPA: {cgpa}"
        for name, roll, branch, cgpa in rows
    ]

    return {
        "status": "success",
        "answer": "Students found:\n" + "\n".join(result_lines)
    }

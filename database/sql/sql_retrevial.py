import sqlite3
import re
import os

# ---------------- PATH ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DB_PATH = os.path.join(PROJECT_ROOT, "students.db")


# ---------------- HELPERS ----------------
def extract_cgpa(query: str):
    q = query.lower()
    patterns = [
        (r"(above|greater than|more than)\s+(\d+(\.\d+)?)", ">"),
        (r"(below|less than)\s+(\d+(\.\d+)?)", "<")
    ]
    for pattern, op in patterns:
        m = re.search(pattern, q)
        if m:
            return op, float(m.group(2))
    return None


def extract_company(query: str):
    """
    Extract company name after keywords like:
    placed in, placed at, placed with
    """
    q = query.lower()
    match = re.search(r"placed\s+(in|at|with)\s+([a-zA-Z]+)", q)
    if match:
        return match.group(2).strip().title()
    return None


def is_count_query(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in ["how many", "count", "number of"])


# ---------------- MAIN HANDLER ----------------
def handle_user_query(query):
    filters = {}
    q = query.lower()

    # -------- Branch --------
    if "cse" in q:
        filters["branch"] = "CSE"
    elif "ece" in q:
        filters["branch"] = "ECE"
    elif "csd" in q:
        filters["branch"] = "CSD"

    # -------- Gender --------
    if "female" in q:
        filters["gender"] = "female"
    elif "male" in q:
        filters["gender"] = "male"

    # -------- CGPA --------
    cgpa_filter = extract_cgpa(q)
    if cgpa_filter:
        filters["cgpa"] = cgpa_filter

    # -------- Placement --------
    company = extract_company(q)
    if "placed" in q:
        filters["placed"] = True
    if company:
        filters["company"] = company

    # -------- Validate --------
    if not filters:
        return {
            "status": "need_more_info",
            "message": "Please specify placement, branch, CGPA, or company."
        }

    # -------- COUNT QUERY --------
    if is_count_query(q):
        sql = "SELECT COUNT(*) FROM students WHERE 1=1"
        params = []
    else:
        sql = "SELECT name, roll_no, branch, cgpa, company_placed FROM students WHERE 1=1"
        params = []

    # -------- Apply Filters --------
    if "branch" in filters:
        sql += " AND branch=?"
        params.append(filters["branch"])

    if "gender" in filters:
        sql += " AND LOWER(gender)=?"
        params.append(filters["gender"])

    if "cgpa" in filters:
        op, val = filters["cgpa"]
        sql += f" AND cgpa {op} ?"
        params.append(val)

    if "placed" in filters:
        sql += " AND company_placed IS NOT NULL"

    if "company" in filters:
        sql += " AND LOWER(company_placed)=?"
        params.append(filters["company"].lower())

    # -------- Execute --------
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()

    # -------- Format Response --------
    if is_count_query(q):
        count = rows[0][0]
        return {
            "status": "success",
            "answer": f"Total students placed in {filters.get('company','companies')}: {count}"
        }

    if not rows:
        return {
            "status": "success",
            "answer": "No students found matching the criteria."
        }

    result = [
        f"{name} ({roll}) - {branch}, CGPA: {cgpa}, Company: {company}"
        for name, roll, branch, cgpa, company in rows
    ]

    return {
        "status": "success",
        "answer": "Placed students:\n" + "\n".join(result)
    }

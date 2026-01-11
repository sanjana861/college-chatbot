import pandas as pd
import sqlite3
import os

# -----------------------------
# PATH SETUP (100% CORRECT)
# -----------------------------
CURRENT_FILE = os.path.abspath(__file__)          # project_bot/database/sql/init_student.py
SQL_DIR = os.path.dirname(CURRENT_FILE)           # project_bot/database/sql
DATABASE_DIR = os.path.dirname(SQL_DIR)            # project_bot/database
PROJECT_ROOT = os.path.dirname(DATABASE_DIR)       # project_bot

DB_PATH = os.path.join(
    PROJECT_ROOT,
    "students.db"
)

EXCEL_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "rawdata",
    "student_dataset.xlsx"
)

# -----------------------------
# SAFETY CHECKS
# -----------------------------
if not os.path.exists(EXCEL_PATH):
    raise FileNotFoundError(f"❌ Excel file not found: {EXCEL_PATH}")

# -----------------------------
# LOAD EXCEL DATA
# -----------------------------
df = pd.read_excel(EXCEL_PATH)

# Normalize column names
df.columns = [
    c.strip().lower().replace(" ", "_")
    for c in df.columns
]

# -----------------------------
# SQLITE SETUP
# -----------------------------
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS students (
    roll_no TEXT PRIMARY KEY,
    name TEXT,
    gender TEXT,
    branch TEXT,
    credits INTEGER,
    cgpa REAL,
    result TEXT,
    degree_name TEXT,
    email_id TEXT,
    joining_year INTEGER,
    passed_year INTEGER,
    admission TEXT,
    company_placed TEXT
);
""")

# -----------------------------
# INSERT DATA
# -----------------------------
for _, row in df.iterrows():
    cur.execute("""
        INSERT OR REPLACE INTO students VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
    """, tuple(row))

conn.commit()
conn.close()

# -----------------------------
# FINAL MESSAGE
# -----------------------------
print("✅ students.db successfully created in project root")
print("📍 DB Path:", DB_PATH)

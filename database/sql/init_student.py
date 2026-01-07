import pandas as pd
import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "students.db")

EXCEL_PATH = os.path.join(
    BASE_DIR, "..", "..", "data", "rawdata", "student_dataset.xlsx"
)
EXCEL_PATH = os.path.normpath(EXCEL_PATH)

df = pd.read_excel(EXCEL_PATH)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

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

for _, r in df.iterrows():
    cur.execute("""
        INSERT OR REPLACE INTO students VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, tuple(r))

conn.commit()
conn.close()

print("✅ students.db created")

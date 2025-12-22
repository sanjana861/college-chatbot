import sqlite3

conn = sqlite3.connect("students.db")
cursor = conn.cursor()

cursor.execute("SELECT name, cgpa FROM students LIMIT 5")
print(cursor.fetchall())

conn.close()

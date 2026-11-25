"""Unit 2 - P10: SQL Query Generation using Structured Prompting (schema + requirement -> SQL, executed on SQLite)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import banner, gemini  # noqa: E402
import re
import sqlite3

banner("U2 P10 - Structured prompt -> SQL -> run on a real SQLite database")
SCHEMA = """CREATE TABLE students(id INTEGER PRIMARY KEY, name TEXT, dept TEXT, cgpa REAL);
CREATE TABLE enrollments(student_id INTEGER, course TEXT, marks INTEGER);"""
db = sqlite3.connect(":memory:"); db.executescript(SCHEMA)
db.executemany("INSERT INTO students VALUES (?,?,?,?)", [(1, "Rohit", "CSE", 8.9), (2, "Asha", "ECE", 9.2), (3, "Bala", "CSE", 7.4), (4, "Divya", "CSE", 9.5)])
db.executemany("INSERT INTO enrollments VALUES (?,?,?)", [(1, "CSA65", 88), (2, "CSA65", 92), (3, "CSA65", 61), (4, "CSA65", 95), (1, "UBA1003", 74)])
TEMPLATE = "ROLE: SQL expert.\nDATABASE: SQLite\nSCHEMA:\n{schema}\nREQUIREMENT: {req}\nOUTPUT: only the SQL query in a sql code block."
for req in ["List CSE students with CGPA above 8, highest first.", "Average marks in CSA65.",
            "Name of the student with the highest marks in CSA65 and their department."]:
    sql = gemini(TEMPLATE.format(schema=SCHEMA, req=req), temperature=0)
    sql = re.search(r"```(?:sql)?\n(.*?)```", sql, re.S).group(1).strip() if "```" in sql else sql.strip()
    print(f"\nRequirement: {req}\nSQL: {sql}")
    try:
        print("Result:", db.execute(sql).fetchall())
    except sqlite3.Error as e:
        print("SQL error:", e)

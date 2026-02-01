import sqlite3
import pandas as pd

conn = sqlite3.connect("data/llm_logs.db")

# 1) Does the column exist?
cols = pd.read_sql_query("PRAGMA table_info(llm_logs)", conn)
print("Columns:", cols["name"].tolist())

# 2) How many total rows?
total = pd.read_sql_query("SELECT COUNT(*) AS n FROM llm_logs", conn)["n"][0]
print("Total rows:", total)

# 3) What experiment_ids exist?
exp = pd.read_sql_query(
    "SELECT COALESCE(experiment_id, '<NULL>') AS experiment_id, COUNT(*) AS n "
    "FROM llm_logs GROUP BY COALESCE(experiment_id, '<NULL>') ORDER BY n DESC",
    conn
)
print(exp)

conn.close()

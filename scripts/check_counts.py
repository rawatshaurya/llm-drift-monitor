import sqlite3

conn = sqlite3.connect("data/llm_logs.db")
cur = conn.cursor()

cur.execute(
    "SELECT model, COUNT(*) FROM llm_logs "
    "WHERE experiment_id = ? "
    "GROUP BY model",
    ("model_switch_equal_samples",)
)

print(cur.fetchall())
conn.close()

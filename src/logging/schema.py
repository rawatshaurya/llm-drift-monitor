import sqlite3

DB_PATH = "data/llm_logs.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS llm_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        experiment_id TEXT,
        model TEXT,
        prompt TEXT,
        response TEXT,
        latency REAL,
        response_length INTEGER,
        refusal_flag INTEGER,
        prompt_embedding BLOB,
        response_embedding BLOB
    )
    """)

    conn.commit()
    conn.close()

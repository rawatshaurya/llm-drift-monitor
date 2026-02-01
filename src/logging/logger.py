import sqlite3
import os
from datetime import datetime
from src.features.embed import embed
from src.features.text_features import response_length, refusal_flag
from src.logging.schema import DB_PATH

def log_interaction(prompt, response, model, latency, experiment_id=None):
    ts = datetime.now().isoformat(timespec="seconds")

    # ✅ read from env unless explicitly provided
    if experiment_id is None:
        experiment_id = os.getenv("EXPERIMENT_ID", "default")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO llm_logs (
        timestamp, experiment_id, model, prompt, response, latency,
        response_length, refusal_flag,
        prompt_embedding, response_embedding
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ts,
        experiment_id,
        model,
        prompt,
        response,
        latency,
        response_length(response),
        refusal_flag(response),
        embed(prompt),
        embed(response)
    ))

    conn.commit()
    conn.close()

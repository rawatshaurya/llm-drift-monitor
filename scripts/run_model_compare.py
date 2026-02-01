import sqlite3
import numpy as np
from src.features.embed import deserialize
from src.drift.detectors import centroid_drift, ks_drift

DB_PATH = "data/llm_logs.db"

MODEL_A = "llama3:8b"
MODEL_B = "qwen2.5:7b"   # change if needed

def fetch_by_model(model_name: str):
    conn = sqlite3.connect(DB_PATH)
    EXPERIMENT_ID = "model_switch_equal_samples"
    cur = conn.cursor()
    cur.execute("""
        SELECT response_embedding, response_length
        FROM llm_logs
        WHERE model = ? AND experiment_id = ?
    """, (model_name, EXPERIMENT_ID))
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return np.empty((0, 384), dtype=np.float32), np.array([], dtype=np.float32)

    emb = np.array([deserialize(r[0]) for r in rows])
    lengths = np.array([r[1] for r in rows], dtype=np.float32)
    return emb, lengths

a_emb, a_len = fetch_by_model(MODEL_A)
b_emb, b_len = fetch_by_model(MODEL_B)

print(f"{MODEL_A}: samples={len(a_len)}")
print(f"{MODEL_B}: samples={len(b_len)}")

if len(a_len) < 10 or len(b_len) < 10:
    print("Not enough samples for one of the models. Log more and rerun.")
    raise SystemExit(0)

print("Semantic drift (centroid cosine distance):", float(centroid_drift(a_emb, b_emb)))
ks_stat, p_val = ks_drift(a_len, b_len)
print("Length KS drift (stat, p):", float(ks_stat), float(p_val))

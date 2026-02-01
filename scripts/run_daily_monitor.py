import sqlite3
import numpy as np
from datetime import datetime, timedelta
from src.features.embed import deserialize
from src.drift.detectors import centroid_drift, ks_drift

DB_PATH = "data/llm_logs.db"
now = datetime.now()

def iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")

today_start = datetime.combine(now.date(), datetime.min.time())
cutoff = now - timedelta(minutes=10)   # last 10 minutes = current

def fetch_window(start_dt, end_dt):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT response_embedding, response_length
        FROM llm_logs
        WHERE timestamp >= ? AND timestamp < ?
    """, (iso(start_dt), iso(end_dt)))
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return np.empty((0, 384), dtype=np.float32), np.array([], dtype=np.float32)

    emb = np.array([deserialize(r[0]) for r in rows])
    lengths = np.array([r[1] for r in rows], dtype=np.float32)
    return emb, lengths

base_emb, base_len = fetch_window(today_start, cutoff)
curr_emb, curr_len = fetch_window(cutoff, now)

print(f"Baseline window: {today_start} → {cutoff} | samples={len(base_len)}")
print(f"Current window : {cutoff} → {now} | samples={len(curr_len)}")

if len(base_len) < 10 or len(curr_len) < 5:
    print("Not enough samples in one window. Log more interactions and rerun.")
    raise SystemExit(0)

print("Output semantic drift (centroid cosine distance):", float(centroid_drift(base_emb, curr_emb)))
ks_stat, p_val = ks_drift(base_len, curr_len)
print("Length KS drift (stat, p):", float(ks_stat), float(p_val))

import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from scipy.spatial.distance import cosine
from scipy.stats import ks_2samp

# ---- Config ----
DB_PATH = "data/llm_logs.db"

st.set_page_config(page_title="LLM Drift Monitor", layout="wide")

# ---- Alert thresholds (tune later) ----
THRESHOLDS = {
    "semantic_warn": 0.08,
    "semantic_alert": 0.15,

    "ks_warn": 0.45,
    "ks_alert": 0.55,

    "p_alert": 0.01,

    "refusal_warn": 0.03,   # 3 percentage points
    "refusal_alert": 0.05,  # 5 percentage points

    "cost_warn": 0.10,      # 10% higher
    "cost_alert": 0.20,     # 20% higher
}

def status_badge(semantic_drift, ks_stat, ks_p, refusal_delta=None, cost_delta=None):
    """
    Returns (label, color, reasons[])
    label: PASS / WARN / ALERT
    """
    reasons = []

    # Handle missing metrics
    if semantic_drift is None or np.isnan(semantic_drift):
        semantic_drift = 0.0
    if ks_stat is None or np.isnan(ks_stat) or ks_p is None or np.isnan(ks_p):
        ks_stat, ks_p = 0.0, 1.0

    # Determine severity
    level = "PASS"

    # Semantic drift
    if semantic_drift >= THRESHOLDS["semantic_alert"]:
        level = "ALERT"
        reasons.append(f"High semantic drift ({semantic_drift:.3f} ≥ {THRESHOLDS['semantic_alert']})")
    elif semantic_drift >= THRESHOLDS["semantic_warn"] and level != "ALERT":
        level = "WARN"
        reasons.append(f"Moderate semantic drift ({semantic_drift:.3f} ≥ {THRESHOLDS['semantic_warn']})")

    # Length drift (stat + significance)
    if ks_stat >= THRESHOLDS["ks_alert"] and ks_p < THRESHOLDS["p_alert"]:
        level = "ALERT"
        reasons.append(f"Strong length drift (KS={ks_stat:.3f}, p={ks_p:.1e})")
    elif ks_stat >= THRESHOLDS["ks_warn"] and ks_p < THRESHOLDS["p_alert"] and level != "ALERT":
        level = "WARN"
        reasons.append(f"Length drift detected (KS={ks_stat:.3f}, p={ks_p:.1e})")

    # Refusal delta (absolute change)
    if refusal_delta is not None and not np.isnan(refusal_delta):
        if refusal_delta >= THRESHOLDS["refusal_alert"]:
            level = "ALERT"
            reasons.append(f"Refusal rate ↑ {refusal_delta:.1%} (≥ {THRESHOLDS['refusal_alert']:.0%})")
        elif refusal_delta >= THRESHOLDS["refusal_warn"] and level != "ALERT":
            level = "WARN"
            reasons.append(f"Refusal rate ↑ {refusal_delta:.1%} (≥ {THRESHOLDS['refusal_warn']:.0%})")

    # Cost delta (relative increase)
    if cost_delta is not None and not np.isnan(cost_delta):
        if cost_delta >= THRESHOLDS["cost_alert"]:
            level = "ALERT"
            reasons.append(f"Estimated cost ↑ {cost_delta:.1%} (≥ {THRESHOLDS['cost_alert']:.0%})")
        elif cost_delta >= THRESHOLDS["cost_warn"] and level != "ALERT":
            level = "WARN"
            reasons.append(f"Estimated cost ↑ {cost_delta:.1%} (≥ {THRESHOLDS['cost_warn']:.0%})")

    color = {"PASS": "green", "WARN": "orange", "ALERT": "red"}[level]
    if not reasons:
        reasons = ["No thresholds exceeded."]
    return level, color, reasons




# ---- Helpers ----
def connect():
    return sqlite3.connect(DB_PATH)

def list_experiments():
    with connect() as conn:
        df = pd.read_sql_query(
            "SELECT COALESCE(experiment_id,'default') AS experiment_id, COUNT(*) AS n "
            "FROM llm_logs GROUP BY COALESCE(experiment_id,'default') ORDER BY n DESC",
            conn,
        )
    return df

TOKEN_PER_WORD = 1.33  # rough English avg
COST_PER_1K_TOKENS = {
    "llama3:8b": 0.0005,     # example placeholder
    "qwen2.5:7b": 0.0004,
}

def estimate_tokens(words: float) -> float:
    return words * TOKEN_PER_WORD

def estimate_cost(tokens: float, model: str) -> float:
    rate = COST_PER_1K_TOKENS.get(model, 0.0005)
    return (tokens / 1000.0) * rate

def render_status(label: str, color: str, reasons: list[str]):
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:12px; margin: 8px 0;">
          <div style="
            padding: 6px 12px;
            border-radius: 999px;
            font-weight: 700;
            color: white;
            background: {color};
            width: fit-content;">
            {label}
          </div>
          <div style="opacity:0.9;">
            {" | ".join(reasons[:2])}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Why this status?"):
        for r in reasons:
            st.write(f"- {r}")


def deserialize(blob: bytes, dim: int = 384) -> np.ndarray:
    # sentence-transformers/all-MiniLM-L6-v2 is 384-d
    return np.frombuffer(blob, dtype=np.float32)

def centroid_cosine_drift(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(cosine(a.mean(axis=0), b.mean(axis=0)))

def drift_over_time(df: pd.DataFrame, window_size: int = 30, step: int = 10):
    """
    Compute rolling drift over time using sliding windows.
    """
    rows = []
    df = df.sort_values("timestamp").reset_index(drop=True)

    for i in range(window_size, len(df), step):
        base = df.iloc[i - window_size:i - step]
        curr = df.iloc[i - step:i]

        if len(base) < 10 or len(curr) < 10:
            continue

        base_emb = np.vstack(base["response_embedding"].apply(deserialize))
        curr_emb = np.vstack(curr["response_embedding"].apply(deserialize))

        sem = centroid_cosine_drift(base_emb, curr_emb)

        ks_stat, p_val = ks_2samp(
            base["response_length"].values,
            curr["response_length"].values
        )

        rows.append({
            "timestamp": curr["timestamp"].iloc[-1],
            "semantic_drift": sem,
            "length_ks": ks_stat,
            "p_value": p_val,
        })

    return pd.DataFrame(rows)


def fetch_logs(start_iso: str | None = None) -> pd.DataFrame:
    with connect() as conn:
        if start_iso:
            q = """
                SELECT timestamp, experiment_id, model, prompt, response, latency, response_length, refusal_flag,
                       response_embedding
                FROM llm_logs
                WHERE timestamp >= ?
                ORDER BY timestamp ASC
            """
            df = pd.read_sql_query(q, conn, params=(start_iso,))
        else:
            q = """
                SELECT timestamp, experiment_id, model, prompt, response, latency, response_length, refusal_flag,
                       response_embedding
                FROM llm_logs
                ORDER BY timestamp ASC
            """
            df = pd.read_sql_query(q, conn)

    if df.empty:
        return df

    # timestamp stored as ISO string like 2026-01-29T20:00:20
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.floor("H")
    return df


def fetch_embeddings_by_filter(where_sql: str, params: tuple) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      embs: (n, 384) float32
      lengths: (n,) float32
    """
    with connect() as conn:
        q = f"""
            SELECT response_embedding, response_length
            FROM llm_logs
            WHERE {where_sql}
        """
        cur = conn.cursor()
        cur.execute(q, params)
        rows = cur.fetchall()

    if not rows:
        return np.empty((0, 384), dtype=np.float32), np.array([], dtype=np.float32)

    embs = np.array([deserialize(r[0]) for r in rows], dtype=np.float32)
    lengths = np.array([r[1] for r in rows], dtype=np.float32)
    return embs, lengths


def drift_snapshot_time_windows(df: pd.DataFrame, baseline_minutes: int, current_minutes: int):
    """
    Compare older vs recent window ending at max timestamp in df.
    """
    if df.empty or df["timestamp"].isna().all():
        return None

    end = df["timestamp"].max()
    current_start = end - timedelta(minutes=current_minutes)
    baseline_start = current_start - timedelta(minutes=baseline_minutes)

    base_emb, base_len = fetch_embeddings_by_filter(
        "timestamp >= ? AND timestamp < ?",
        (baseline_start.isoformat(timespec="seconds"), current_start.isoformat(timespec="seconds")),
    )
    curr_emb, curr_len = fetch_embeddings_by_filter(
        "timestamp >= ? AND timestamp <= ?",
        (current_start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")),
    )

    out = {
        "baseline_start": baseline_start,
        "baseline_end": current_start,
        "current_start": current_start,
        "current_end": end,
        "n_baseline": int(len(base_len)),
        "n_current": int(len(curr_len)),
    }

    if len(base_len) >= 5 and len(curr_len) >= 5:
        out["semantic_drift"] = centroid_cosine_drift(base_emb, curr_emb)
        ks_stat, p_val = ks_2samp(base_len, curr_len)
        out["ks_stat"] = float(ks_stat)
        out["ks_p"] = float(p_val)
    else:
        out["semantic_drift"] = float("nan")
        out["ks_stat"] = float("nan")
        out["ks_p"] = float("nan")

    return out


def model_compare(model_a: str, model_b: str, experiment_id: str, k: int):
    def fetch_last_k(model_name: str):
        with connect() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT response_embedding, response_length
                FROM llm_logs
                WHERE model = ? AND COALESCE(experiment_id,'default') = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (model_name, experiment_id, k))
            rows = cur.fetchall()

        if not rows:
            return np.empty((0, 384), dtype=np.float32), np.array([], dtype=np.float32)

        embs = np.array([deserialize(r[0]) for r in rows], dtype=np.float32)
        lengths = np.array([r[1] for r in rows], dtype=np.float32)
        return embs, lengths

    a_emb, a_len = fetch_last_k(model_a)
    b_emb, b_len = fetch_last_k(model_b)

    out = {"n_a": int(len(a_len)), "n_b": int(len(b_len))}

    if len(a_len) >= 10 and len(b_len) >= 10:
        out["semantic_drift"] = centroid_cosine_drift(a_emb, b_emb)
        ks_stat, p_val = ks_2samp(a_len, b_len)
        out["ks_stat"] = float(ks_stat)
        out["ks_p"] = float(p_val)
    else:
        out["semantic_drift"] = float("nan")
        out["ks_stat"] = float("nan")
        out["ks_p"] = float("nan")

    return out



# ---- UI ----
st.title("LLM Drift Monitor Dashboard")

# Sidebar controls
st.sidebar.header("Controls")

lookback_days = st.sidebar.slider("Lookback (days)", min_value=1, max_value=30, value=7)
baseline_minutes = st.sidebar.slider("Baseline window (minutes)", 30, 12 * 60, 8 * 60, step=30)
current_minutes = st.sidebar.slider("Current window (minutes)", 10, 6 * 60, 60, step=10)

start_dt = datetime.now() - timedelta(days=lookback_days)
start_iso = start_dt.isoformat(timespec="seconds")



exp_df = list_experiments()
exp_options = exp_df["experiment_id"].tolist()
selected_exp = st.sidebar.selectbox("Experiment", exp_options, index=0)

# Equal-sample size per model
K = st.sidebar.slider("Equal samples per model (K)", 10, 200, 93, step=1)


df = fetch_logs(start_iso)

df = df[df["experiment_id"].fillna("default") == selected_exp].copy()

if df.empty:
    st.warning("No logs found in the selected lookback window. Run scripts.run_prompt to generate data.")
    st.stop()

# Top KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total logs", f"{len(df)}")
c2.metric("Models seen", f"{df['model'].nunique()}")
c3.metric("Avg response length", f"{df['response_length'].mean():.1f} words")
c4.metric("Avg latency", f"{df['latency'].mean():.2f} s")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["Drift Snapshot", "Trends", "Model Compare", "Recent Logs"])

with tab1:
    st.subheader("Drift snapshot (time windows)")
    snap = drift_snapshot_time_windows(df, baseline_minutes=baseline_minutes, current_minutes=current_minutes)

    if snap is None:
        st.info("Not enough data to compute drift snapshot.")
    else:
        st.write(
            f"Baseline: **{snap['baseline_start']} → {snap['baseline_end']}** "
            f"(n={snap['n_baseline']})  \n"
            f"Current: **{snap['current_start']} → {snap['current_end']}** "
            f"(n={snap['n_current']})"
        )

        k1, k2, k3 = st.columns(3)
        k1.metric("Semantic drift (centroid cosine)", "—" if np.isnan(snap["semantic_drift"]) else f"{snap['semantic_drift']:.4f}")
        k2.metric("Length KS statistic", "—" if np.isnan(snap["ks_stat"]) else f"{snap['ks_stat']:.4f}")
        k3.metric("KS p-value", "—" if np.isnan(snap["ks_p"]) else f"{snap['ks_p']:.2e}")

        st.caption(
            "Interpretation tip: semantic drift ~0.00–0.03 tiny, 0.03–0.08 mild, 0.08–0.15 moderate, >0.15 high. "
            "KS p-value < 0.01 indicates significant length distribution change."
        )
        label, color, reasons = status_badge(
            snap.get("semantic_drift", np.nan),
            snap.get("ks_stat", np.nan),
            snap.get("ks_p", np.nan),
        )

        st.markdown("### Status")
        render_status(label, color, reasons)

with tab2:
    st.subheader("Trends over time")

    # Hourly trends
    hourly = (
        df.groupby(["hour", "model"], as_index=False)
        .agg(
            n=("model", "count"),
            avg_len=("response_length", "mean"),
            avg_latency=("latency", "mean"),
            refusal_rate=("refusal_flag", "mean"),
        )
        .sort_values("hour")
    )

    st.markdown("### Avg response length (hourly)")
    st.line_chart(hourly.pivot(index="hour", columns="model", values="avg_len"))

    st.markdown("### Avg latency (hourly)")
    st.line_chart(hourly.pivot(index="hour", columns="model", values="avg_latency"))

    st.markdown("### Refusal rate (hourly)")
    st.line_chart(hourly.pivot(index="hour", columns="model", values="refusal_rate"))
    st.markdown("### Drift over time")

    drift_df = drift_over_time(df, window_size=40, step=10)

    if drift_df.empty:
        st.info("Not enough data to compute drift over time.")
    else:
        st.line_chart(
            drift_df.set_index("timestamp")[["semantic_drift", "length_ks"]],
            height=300
        )

        st.caption(
            "Semantic drift tracks meaning changes; Length KS tracks verbosity/style drift. "
            "Sustained spikes indicate behavior change."
        )





with tab3:
    st.subheader("Model vs Model comparison (same prompt set)")

    models = sorted(df["model"].dropna().unique().tolist())
    if len(models) < 2:
        st.info("Need at least 2 different models logged to compare. Log with another model and come back.")
    else:
        colA, colB = st.columns(2)
        with colA:
            model_a = st.selectbox("Model A", models, index=0)
        with colB:
            model_b = st.selectbox("Model B", models, index=1)

        comp = model_compare(model_a, model_b, selected_exp, K)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Samples (A)", f"{comp['n_a']}")
        c2.metric("Samples (B)", f"{comp['n_b']}")
        c3.metric("Semantic drift", "—" if np.isnan(comp["semantic_drift"]) else f"{comp['semantic_drift']:.4f}")
        c4.metric("Length KS (p)", "—" if np.isnan(comp["ks_stat"]) else f"{comp['ks_stat']:.3f} ({comp['ks_p']:.2e})")

        st.markdown("### Aggregates by model (within lookback)")
        agg = (
            df.groupby("model", as_index=False)
            .agg(
                n=("model", "count"),
                avg_len=("response_length", "mean"),
                p95_len=("response_length", lambda x: x.quantile(0.95)),
                avg_latency=("latency", "mean"),
                refusal_rate=("refusal_flag", "mean"),
            )
            .sort_values("n", ascending=False)
        )
        st.dataframe(agg, use_container_width=True)

        # --- Compute deltas for badge (Model B vs Model A) ---
        def get_val(model_name: str, col: str):
            return float(agg.loc[agg["model"] == model_name, col].iloc[0])

        # Refusal delta (absolute)
        refusal_a = get_val(model_a, "refusal_rate")
        refusal_b = get_val(model_b, "refusal_rate")
        refusal_delta = max(0.0, refusal_b - refusal_a)  # focus on "increase"

        # Estimated cost delta from avg_len (relative)
        tokens_a = estimate_tokens(get_val(model_a, "avg_len"))
        tokens_b = estimate_tokens(get_val(model_b, "avg_len"))
        cost_a = estimate_cost(tokens_a, model_a)
        cost_b = estimate_cost(tokens_b, model_b)
        cost_delta = (cost_b - cost_a) / cost_a if cost_a > 0 else np.nan

        label, color, reasons = status_badge(
            comp.get("semantic_drift", np.nan),
            comp.get("ks_stat", np.nan),
            comp.get("ks_p", np.nan),
            refusal_delta=refusal_delta,
            cost_delta=cost_delta,
        )

        st.markdown("### Status")
        render_status(label, color, reasons)

        st.markdown("### Estimated cost impact (per response)")

        cost_df = agg.copy()
        cost_df["est_tokens"] = cost_df["avg_len"].apply(estimate_tokens)
        cost_df["est_cost_usd"] = cost_df.apply(
            lambda r: estimate_cost(r["est_tokens"], r["model"]), axis=1
        )

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                f"{model_a} avg cost",
                f"${cost_df[cost_df.model == model_a]['est_cost_usd'].iloc[0]:.5f}"
            )

        with c2:
            st.metric(
                f"{model_b} avg cost",
                f"${cost_df[cost_df.model == model_b]['est_cost_usd'].iloc[0]:.5f}"
            )

        st.dataframe(
            cost_df[["model", "avg_len", "est_tokens", "est_cost_usd"]],
            use_container_width=True
        )

        st.caption(
            "Estimated using avg response length × 1.33 tokens/word. "
            "Even small verbosity drift compounds at scale."
        )

with tab4:
    st.subheader("Recent logs")
    show_cols = ["timestamp", "model", "response_length", "latency", "refusal_flag", "prompt"]
    st.dataframe(df.sort_values("timestamp", ascending=False)[show_cols].head(50), use_container_width=True)

    with st.expander("Show full response text (latest 5)"):
        latest = df.sort_values("timestamp", ascending=False).head(5)
        for _, row in latest.iterrows():
            st.markdown(f"**{row['timestamp']} — {row['model']}**  \nPrompt: {row['prompt']}")
            st.code(str(row["response"])[:4000])
            st.divider()

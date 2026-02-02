# 🚦 LLM Drift Monitor  
### Semantic • Structural • Safety • Cost Drift Monitoring for LLMs

A **production-style LLM behavior monitoring system** that answers the question:

> **“Is this model still behaving the way we expect?”**

This project goes beyond accuracy to detect **semantic drift**, **verbosity drift**, **safety drift**, and **cost drift** in Large Language Models using statistical tests, embeddings, and controlled experiments.

---

## ✨ Key Features

- 🔍 **Semantic Drift Detection**
  - Embedding centroid cosine distance on model responses
- 📏 **Structural Drift Detection**
  - Response length distribution shift via Kolmogorov–Smirnov test
- 🛡️ **Safety Drift**
  - Refusal-rate changes across models or time
- 💰 **Estimated Cost Drift**
  - Token + cost estimation from verbosity changes
- 🧪 **Controlled Experiments**
  - Equal-sample A/B comparisons using `EXPERIMENT_ID`
- 📈 **Drift Over Time**
  - Rolling-window drift visualization
- 🚦 **Auto Alerts**
  - PASS / WARN / ALERT badges with explainable reasons
- 📊 **Streamlit Dashboard**
  - Interactive, production-style monitoring UI

---

## 🧱 Tech Stack

- **LLM Runtime:** Ollama (local inference)
- **Models Tested:** LLaMA-3-8B, Qwen-2.5-7B
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2
- **Storage:** SQLite
- **Statistics:** SciPy (KS test)
- **Visualization:** Streamlit
- **Language:** Python 3.10+

---

## 📂 Repository Structure
```
llm-drift-monitor/
├─ dashboard/
│ └─ app.py
├─ data/
│ └─ llm_logs.db # ignored by git
├─ reports/
│ └─ .gitkeep
├─ scripts/
│ ├─ run_prompt.py
│ ├─ run_daily_monitor.py
│ ├─ run_model_compare.py
│ ├─ check_counts.py
│ └─ inspect_experiments.py
├─ src/
│ ├─ drift/
│ │ └─ detectors.py
│ ├─ features/
│ │ ├─ embed.py
│ │ └─ text_features.py
│ ├─ llm/
│ │ └─ ollama_client.py
│ └─ logging/
│ ├─ logger.py
│ └─ schema.py
├─ .gitignore
├─ LICENSE
├─ requirements.txt
└─ README.md

```

---

## 🚀 Quickstart

### 1️⃣ Create & activate virtual environment (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
### 2️⃣ Install dependencies

pip install -r requirements.txt
### 3️⃣ Start Ollama & pull models

ollama list
ollama pull llama3:8b
ollama pull qwen2.5:7b
### 4️⃣ Log LLM interactions
This logs prompts, responses, embeddings, latency, length, and refusal flags.


python -m scripts.run_prompt
### 5️⃣ Run CLI drift monitor

python -m scripts.run_daily_monitor
### 6️⃣ Launch Streamlit dashboard

streamlit run dashboard/app.py

```
🧪 Controlled Model Switch Experiment (Equal Samples)
This performs a clean A/B test:

-Same prompts
-Same sample size
-Only model changes

```
$env:EXPERIMENT_ID="model_switch_equal_samples"

$env:OLLAMA_MODEL="llama3:8b"
1..31 | % { python -m scripts.run_prompt }   # ≈93 samples

$env:OLLAMA_MODEL="qwen2.5:7b"
1..31 | % { python -m scripts.run_prompt }

python -m scripts.check_counts
python -m scripts.run_model_compare

```
Expected outcome:

✅ Mild semantic drift

⚠️ Strong verbosity / length drift

⚠️ Possible refusal-rate changes

📊 Dashboard Overview
Tabs included:
-Drift Snapshot – current baseline vs recent behavior
-Drift Over Time – rolling-window semantic & length drift
-Model Compare – equal-sample A/B comparison
-Recent Logs – raw interaction inspection

Auto Alert System
Each comparison is labeled:

-🟢 PASS – normal variation
-🟡 WARN – moderate drift detected
-🔴 ALERT – statistically significant behavior change

Alerts are triggered using:

-semantic drift thresholds
-KS statistic + p-value
-refusal-rate increase
-estimated cost increase

📐 Metrics & Interpretation
Semantic Drift (Cosine Distance)
Range	Meaning
0.00 – 0.03	Tiny
0.03 – 0.08	Mild
0.08 – 0.15	Moderate
> 0.15	High

Structural Drift (KS Test)
-KS statistic ↑ → larger distribution shift
-p < 0.01 → statistically significant drift

Why this matters
Even when semantic content stays stable, drift in:

-verbosity
-latency
-refusal behavior
-token cost

can silently degrade UX, increase spend, or break workflows.

🧠 Key Insight Demonstrated
Switching from LLaMA-3-8B to Qwen-2.5-7B caused major verbosity and refusal-rate drift while semantic meaning remained largely stable — showing why accuracy alone is insufficient for LLM evaluation.

🔒 Notes
-data/llm_logs.db is intentionally not committed
-Use EXPERIMENT_ID to keep experiments isolated and reproducible
-Cost estimates are approximate (token ≈ 1.33 × words)


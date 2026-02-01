from src.llm.ollama_client import generate
from src.logging.logger import log_interaction
from src.logging.schema import init_db

init_db()

PROMPTS = [
    "What are the prerequisites for DMS440?",
    "Explain backpropagation in simple terms.",
    "Can I take a graduate ML course without linear algebra?"
]

for p in PROMPTS:
    out = generate(p)
    log_interaction(
        prompt=p,
        response=out["response"],
        model=out["model"],
        latency=out["latency"]
    )
    print("Logged:", p)

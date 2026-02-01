## In order to run any model other than the default, use -> $env:OLLAMA_MODEL="model_name"


import time
import os
import ollama

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")

def generate(prompt: str, model: str = DEFAULT_MODEL):
    start = time.time()
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    latency = time.time() - start
    text = response["message"]["content"]
    return {"response": text, "latency": latency, "model": model}

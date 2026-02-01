import numpy as np
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text: str) -> bytes:
    vec = _model.encode(text)
    return np.asarray(vec, dtype=np.float32).tobytes()

def deserialize(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)

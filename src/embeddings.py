import base64

import httpx
import numpy as np

from .utils import SqliteCache, make_key


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class Embedder:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_s: float = 300.0,
        cache: SqliteCache | None = None,
        batch_size: int = 32,
        query_prefix: str = "",
        document_prefix: str = "",
    ):
        self.model = model
        self.name = f"ollama:{model}"
        self.url = f"{base_url.rstrip('/')}/api/embed"
        self.cache = cache
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self._client = httpx.Client(timeout=timeout_s)

    def _embed(self, texts: list[str]) -> np.ndarray:
        response = self._client.post(self.url, json={"model": self.model, "input": texts})
        response.raise_for_status()
        return np.asarray(response.json()["embeddings"], dtype=np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[np.ndarray | None] = [None] * len(texts)
        missing: list[int] = []
        for position, text in enumerate(texts):
            stored = self.cache.get(make_key(self.name, text)) if self.cache else None
            if stored is None:
                missing.append(position)
            else:
                vectors[position] = np.frombuffer(base64.b64decode(stored["v"]), dtype=np.float32)
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            computed = normalize_rows(self._embed([texts[i] for i in batch]))
            for position, vector in zip(batch, computed):
                vectors[position] = vector
                if self.cache:
                    self.cache.put(make_key(self.name, texts[position]), {"v": base64.b64encode(vector.tobytes()).decode()})
        return np.vstack(vectors).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([self.query_prefix + text])[0]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.embed([self.document_prefix + text for text in texts])

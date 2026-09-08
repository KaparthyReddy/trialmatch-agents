"""
Adapts Ollama's embeddings endpoint to the synchronous callable interface
ChromaDB expects for a custom embedding function.

Uses /api/embed (current) rather than /api/embeddings (legacy, removed in
newer Ollama versions) — the current endpoint takes an "input" field and
returns {"embeddings": [[...]]}  (plural, list of vectors) even for a
single text, rather than the old singular "embedding" key.

Deliberately synchronous rather than wrapping the async OllamaClient with
asyncio.run() — ChromaDB invokes this from code that may already be
running inside an active asyncio event loop, and asyncio.run() cannot be
called from within an already-running loop.
"""

import os
import httpx
from chromadb.api.types import Documents, Embeddings


class OllamaEmbeddingFunction:

    def __init__(self, base_url: str | None = None, embed_model: str | None = None, timeout: float = 60.0):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.embed_model = embed_model or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self.timeout = timeout

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed_sync(text) for text in input]

    def _embed_sync(self, text: str) -> list[float]:
        payload = {"model": self.embed_model, "input": text}
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.base_url}/api/embed", json=payload)
            response.raise_for_status()
            return response.json()["embeddings"][0]

    def name(self) -> str:
        return "ollama-embedding-function"
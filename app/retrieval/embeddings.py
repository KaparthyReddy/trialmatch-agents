"""
Adapts OllamaClient's embedding call to the callable interface ChromaDB
expects for a custom embedding function.
"""

from chromadb.api.types import Documents, Embeddings
import asyncio

from app.llm.ollama_client import OllamaClient


class OllamaEmbeddingFunction:
    """ChromaDB calls embedding functions synchronously, so this wraps the
    async OllamaClient call with asyncio.run under the hood."""

    def __init__(self, ollama_client: OllamaClient):
        self.ollama_client = ollama_client

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed_sync(text) for text in input]

    def _embed_sync(self, text: str) -> list[float]:
        return asyncio.run(self.ollama_client.embed(text))

    def name(self) -> str:
        return "ollama-embedding-function"

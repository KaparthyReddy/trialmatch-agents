"""
Thin wrapper around Ollama's REST API. Deliberately not using the `ollama`
pip package's global client, since that reads its host from a process-wide
env var — this class takes an explicit base URL so it works cleanly in
Docker (where the app and Ollama are different containers) and locally.
"""

import os
import httpx


class OllamaClient:

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 embed_model: str | None = None, timeout: float = 60.0):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1")
        self.embed_model = embed_model or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self.timeout = timeout

    async def generate(self, prompt: str, system: str | None = None) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            return response.json()["response"]

    async def embed(self, text: str) -> list[float]:
        payload = {"model": self.embed_model, "prompt": text}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/api/embeddings", json=payload)
            response.raise_for_status()
            return response.json()["embedding"]

    async def is_reachable(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except httpx.HTTPError:
            return False

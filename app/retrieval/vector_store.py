"""
ChromaDB-backed vector store for trial summaries. Trials pulled from
ClinicalTrials.gov are embedded and stored here so the SynthesisAgent can
retrieve the most relevant ones for a given query via similarity search,
rather than just dumping every fetched trial into the LLM's context.
"""

import os
import chromadb

from app.models.schemas import TrialResult
from app.retrieval.embeddings import OllamaEmbeddingFunction
from app.llm.ollama_client import OllamaClient


class TrialVectorStore:

    def __init__(self, persist_dir: str | None = None, ollama_client: OllamaClient | None = None):
        self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.embedding_fn = OllamaEmbeddingFunction(ollama_client or OllamaClient())
        self.collection = self.client.get_or_create_collection(
            name="clinical_trials",
            embedding_function=self.embedding_fn,
        )

    def add_trials(self, trials: list[TrialResult]) -> None:
        if not trials:
            return

        self.collection.upsert(
            ids=[t.nct_id for t in trials],
            documents=[f"{t.title}\n{t.summary}" for t in trials],
            metadatas=[{
                "title": t.title,
                "status": t.status,
                "conditions": ", ".join(t.conditions),
                "url": t.url,
            } for t in trials],
        )

    def query_similar(self, query_text: str, n_results: int = 3) -> list[dict]:
        results = self.collection.query(query_texts=[query_text], n_results=n_results)

        matches = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for nct_id, document, metadata in zip(ids, documents, metadatas):
            matches.append({"nct_id": nct_id, "document": document, "metadata": metadata})

        return matches

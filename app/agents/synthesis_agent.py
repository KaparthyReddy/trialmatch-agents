"""
Final agent in the graph: retrieves the most relevant indexed trials for
the query via vector similarity search, combines them with any
interaction warnings, and asks the LLM to produce one coherent, cited
answer. This is the step that turns "here's a pile of data" into "here's
an answer" — the actual point of the whole pipeline.
"""

from app.agents.base_agent import BaseAgent, AgentState
from app.retrieval.vector_store import TrialVectorStore
from app.llm.ollama_client import OllamaClient

SYSTEM_PROMPT = (
    "You are a clinical research assistant. You are given retrieved clinical "
    "trial summaries and any known drug-interaction warnings. Write a clear, "
    "cautious answer to the user's question. Always cite trial NCT IDs when "
    "referencing a specific trial. Always mention interaction warnings "
    "prominently if any are present. Do not give direct medical advice — "
    "frame findings as information to discuss with a clinician."
)


class SynthesisAgent(BaseAgent):
    name = "synthesis_agent"

    def __init__(self, vector_store: TrialVectorStore, ollama_client: OllamaClient):
        self.vector_store = vector_store
        self.ollama_client = ollama_client

    async def run(self, state: AgentState) -> AgentState:
        query = state.get("query", "")
        warnings = state.get("interaction_warnings", [])

        relevant = self.vector_store.query_similar(query, n_results=3)

        trial_context = "\n\n".join(
            f"[{match['nct_id']}] {match['metadata'].get('title', '')}\n{match['document']}"
            for match in relevant
        ) or "No matching trials were retrieved."

        warning_context = "\n".join(
            f"- {w.drug_a} + {w.drug_b} ({w.severity}): {w.note}"
            for w in warnings
        ) or "No known drug interactions flagged."

        prompt = (
            f"User question: {query}\n\n"
            f"Retrieved trials:\n{trial_context}\n\n"
            f"Interaction warnings:\n{warning_context}\n\n"
            "Write the final answer now."
        )

        answer = await self.ollama_client.generate(prompt, system=SYSTEM_PROMPT)
        state["answer"] = answer
        return state

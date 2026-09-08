"""
Final agent in the graph: retrieves the most relevant indexed trials for
the query via vector similarity search, combines them with any
interaction warnings, and asks the LLM to produce a coherent, cited
answer.

Interaction warnings are deterministically prepended to the final answer
rather than left entirely to the LLM to surface. This is a deliberate
safety-critical design choice: an LLM can be instructed to mention
warnings, but instruction-following isn't guaranteed, and drug
interaction information is exactly the category of output where
"probably mentioned it" isn't an acceptable reliability bar. The LLM's
narrative still receives the warnings as context and is asked to weave
them in naturally, but the raw warning data is never solely dependent on
the model choosing to repeat it.
"""

from app.agents.base_agent import BaseAgent, AgentState
from app.retrieval.vector_store import TrialVectorStore
from app.llm.ollama_client import OllamaClient

SYSTEM_PROMPT = (
    "You are a clinical research assistant. You are given retrieved clinical "
    "trial summaries and any known drug-interaction warnings. Write a clear, "
    "cautious narrative answer to the user's question. Always cite trial NCT "
    "IDs when referencing a specific trial. Do not give direct medical advice "
    "— frame findings as information to discuss with a clinician. Note: any "
    "drug interaction warnings will be displayed separately and verbatim "
    "above your answer, so you do not need to restate them in full, but you "
    "may reference them briefly if relevant to your narrative."
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
            f"Interaction warnings (already shown to the user separately):\n{warning_context}\n\n"
            "Write the narrative answer now."
        )

        llm_answer = await self.ollama_client.generate(prompt, system=SYSTEM_PROMPT)
        state["answer"] = self._compose_final_answer(warnings, llm_answer)
        return state

    @staticmethod
    def _compose_final_answer(warnings, llm_answer: str) -> str:
        if not warnings:
            return llm_answer

        warning_lines = "\n".join(
            f"- **{w.drug_a} + {w.drug_b}** ({w.severity}): {w.note}"
            for w in warnings
        )
        banner = f"⚠️ **Drug Interaction Warning(s) Detected**\n{warning_lines}\n\n---\n\n"
        return banner + llm_answer
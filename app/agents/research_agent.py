"""
Queries the real ClinicalTrials.gov API for trials matching the user's
query, then indexes the results into the vector store so the
SynthesisAgent can retrieve the most relevant subset later rather than
re-reading every fetched trial from scratch.
"""

from app.agents.base_agent import BaseAgent, AgentState
from app.clients.clinical_trials_client import ClinicalTrialsClient
from app.retrieval.vector_store import TrialVectorStore


class ResearchAgent(BaseAgent):
    name = "research_agent"

    def __init__(self, trials_client: ClinicalTrialsClient, vector_store: TrialVectorStore):
        self.trials_client = trials_client
        self.vector_store = vector_store

    async def run(self, state: AgentState) -> AgentState:
        query = state.get("query", "")
        trials = await self.trials_client.search_trials(query, max_results=5)

        self.vector_store.add_trials(trials)

        state["trials"] = trials
        return state

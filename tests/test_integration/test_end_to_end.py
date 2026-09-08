"""
Real end-to-end integration test: hits the live ClinicalTrials.gov API and
a real local Ollama instance, exercising the full agent graph exactly as
a production request would. Skipped by default since it needs Ollama
running locally with the required models pulled - CI has neither, so this
intentionally does not run there.

Run manually with:
    RUN_INTEGRATION_TESTS=1 pytest tests/test_integration -v
"""

import os
import pytest

from app.llm.ollama_client import OllamaClient
from app.retrieval.vector_store import TrialVectorStore
from app.clients.clinical_trials_client import ClinicalTrialsClient
from app.clients.drug_interaction_client import DrugInteractionClient
from app.agents.research_agent import ResearchAgent
from app.agents.interaction_agent import InteractionAgent
from app.agents.synthesis_agent import SynthesisAgent
from app.graph.orchestrator import TrialMatchOrchestrator

pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_INTEGRATION_TESTS"),
    reason="Requires a live Ollama instance with models pulled; set RUN_INTEGRATION_TESTS=1 to run"
)


@pytest.mark.asyncio
async def test_full_pipeline_against_real_apis():
    ollama_client = OllamaClient()
    vector_store = TrialVectorStore(ollama_client=ollama_client)
    trials_client = ClinicalTrialsClient()
    interaction_client = DrugInteractionClient()

    orchestrator = TrialMatchOrchestrator(
        ResearchAgent(trials_client, vector_store),
        InteractionAgent(interaction_client),
        SynthesisAgent(vector_store, ollama_client),
    )

    result = await orchestrator.run_query(
        query="clinical trials for stage 2 lung cancer",
        medications=["warfarin", "aspirin"],
    )

    assert len(result["trials"]) > 0
    assert any(t.nct_id.startswith("NCT") for t in result["trials"])

    assert len(result["interaction_warnings"]) == 1
    assert result["interaction_warnings"][0].severity == "MAJOR"

    assert "Drug Interaction Warning" in result["answer"]
    assert "warfarin" in result["answer"].lower()
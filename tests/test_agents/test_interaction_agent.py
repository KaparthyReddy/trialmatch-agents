import pytest

from app.agents.interaction_agent import InteractionAgent
from app.clients.drug_interaction_client import DrugInteractionClient
from app.models.schemas import TrialResult


@pytest.fixture
def agent():
    return InteractionAgent(DrugInteractionClient())


@pytest.mark.asyncio
async def test_flags_interaction_in_patient_medications(agent):
    state = {"query": "test", "medications": ["warfarin", "aspirin"], "trials": []}
    result = await agent.run(state)

    assert len(result["interaction_warnings"]) == 1
    assert result["interaction_warnings"][0].severity == "MAJOR"


@pytest.mark.asyncio
async def test_no_warnings_for_safe_medications(agent):
    state = {"query": "test", "medications": ["paracetamol"], "trials": []}
    result = await agent.run(state)

    assert result["interaction_warnings"] == []


@pytest.mark.asyncio
async def test_flags_interaction_mentioned_in_trial_summary(agent):
    trial = TrialResult(
        nct_id="NCT123", title="Test", status="RECRUITING",
        conditions=["Test"], summary="This trial involves aspirin as a comparator.",
        url="https://example.com",
    )
    state = {"query": "test", "medications": ["warfarin"], "trials": [trial]}
    result = await agent.run(state)

    assert len(result["interaction_warnings"]) >= 1
    assert "NCT123" in result["interaction_warnings"][0].note

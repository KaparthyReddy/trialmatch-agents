import pytest
from unittest.mock import AsyncMock, MagicMock

from app.agents.research_agent import ResearchAgent
from app.models.schemas import TrialResult


@pytest.mark.asyncio
async def test_research_agent_fetches_and_indexes_trials():
    mock_trial = TrialResult(
        nct_id="NCT999", title="Mock Trial", status="RECRUITING",
        conditions=["Test Condition"], summary="Summary text.",
        url="https://example.com",
    )

    mock_trials_client = MagicMock()
    mock_trials_client.search_trials = AsyncMock(return_value=[mock_trial])

    mock_vector_store = MagicMock()

    agent = ResearchAgent(mock_trials_client, mock_vector_store)
    state = {"query": "test condition", "medications": []}

    result = await agent.run(state)

    assert result["trials"] == [mock_trial]
    mock_vector_store.add_trials.assert_called_once_with([mock_trial])

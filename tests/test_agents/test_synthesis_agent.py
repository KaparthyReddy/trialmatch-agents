import pytest
from unittest.mock import AsyncMock, MagicMock

from app.agents.synthesis_agent import SynthesisAgent


@pytest.mark.asyncio
async def test_synthesis_agent_calls_llm_with_context():
    mock_vector_store = MagicMock()
    mock_vector_store.query_similar.return_value = [
        {"nct_id": "NCT111", "document": "Trial doc text", "metadata": {"title": "A Trial"}}
    ]

    mock_ollama_client = MagicMock()
    mock_ollama_client.generate = AsyncMock(return_value="This is the synthesized answer.")

    agent = SynthesisAgent(mock_vector_store, mock_ollama_client)
    state = {"query": "lung cancer trials", "interaction_warnings": [], "trials": []}

    result = await agent.run(state)

    assert result["answer"] == "This is the synthesized answer."
    mock_ollama_client.generate.assert_called_once()
    call_args = mock_ollama_client.generate.call_args
    assert "NCT111" in call_args.args[0]

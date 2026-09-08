import pytest
from unittest.mock import AsyncMock, MagicMock

from app.graph.orchestrator import TrialMatchOrchestrator


@pytest.mark.asyncio
async def test_orchestrator_runs_agents_in_order():
    call_order = []

    async def research_run(state):
        call_order.append("research")
        state["trials"] = []
        return state

    async def interaction_run(state):
        call_order.append("interaction")
        state["interaction_warnings"] = []
        return state

    async def synthesis_run(state):
        call_order.append("synthesis")
        state["answer"] = "final answer"
        return state

    research_agent = MagicMock(name="research_agent")
    research_agent.name = "research_agent"
    research_agent.run = AsyncMock(side_effect=research_run)

    interaction_agent = MagicMock(name="interaction_agent")
    interaction_agent.name = "interaction_agent"
    interaction_agent.run = AsyncMock(side_effect=interaction_run)

    synthesis_agent = MagicMock(name="synthesis_agent")
    synthesis_agent.name = "synthesis_agent"
    synthesis_agent.run = AsyncMock(side_effect=synthesis_run)

    orchestrator = TrialMatchOrchestrator(research_agent, interaction_agent, synthesis_agent)
    result = await orchestrator.run_query("test query", ["aspirin"])

    assert call_order == ["research", "interaction", "synthesis"]
    assert result["answer"] == "final answer"

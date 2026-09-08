"""
Wires the three agents into an explicit LangGraph state machine. This is
the actual "agentic" part of the system: rather than one long prompt
pretending to be multiple steps, this is a real graph where each node is
an independent agent, and the state is threaded through explicitly.

Current flow is linear (research -> interaction -> synthesis) because
InteractionAgent checks trial summaries for mentioned drugs, so it needs
ResearchAgent's output first. A natural extension point: branch research
and a "safety pre-check" in parallel, then fan back in before synthesis.
"""

from langgraph.graph import StateGraph, END

from app.agents.base_agent import AgentState
from app.agents.research_agent import ResearchAgent
from app.agents.interaction_agent import InteractionAgent
from app.agents.synthesis_agent import SynthesisAgent


def build_graph(research_agent: ResearchAgent, interaction_agent: InteractionAgent,
                 synthesis_agent: SynthesisAgent):
    graph = StateGraph(AgentState)

    graph.add_node(research_agent.name, research_agent.run)
    graph.add_node(interaction_agent.name, interaction_agent.run)
    graph.add_node(synthesis_agent.name, synthesis_agent.run)

    graph.set_entry_point(research_agent.name)
    graph.add_edge(research_agent.name, interaction_agent.name)
    graph.add_edge(interaction_agent.name, synthesis_agent.name)
    graph.add_edge(synthesis_agent.name, END)

    return graph.compile()


class TrialMatchOrchestrator:
    """Thin façade so the API layer doesn't need to know about LangGraph
    specifics — it just calls `run_query`."""

    def __init__(self, research_agent: ResearchAgent, interaction_agent: InteractionAgent,
                 synthesis_agent: SynthesisAgent):
        self.compiled_graph = build_graph(research_agent, interaction_agent, synthesis_agent)

    async def run_query(self, query: str, medications: list[str]) -> AgentState:
        initial_state: AgentState = {
            "query": query,
            "medications": medications,
            "trials": [],
            "interaction_warnings": [],
            "answer": "",
        }
        final_state = await self.compiled_graph.ainvoke(initial_state)
        return final_state

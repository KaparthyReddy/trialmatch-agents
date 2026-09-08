"""
Shared interface every agent implements, plus the typed state object that
flows through the LangGraph orchestration. Defining this once means the
graph doesn't need to know anything about each agent's internals — it
just calls `run(state)` and gets an updated state back.
"""

from abc import ABC, abstractmethod
from typing import TypedDict

from app.models.schemas import TrialResult, InteractionWarning


class AgentState(TypedDict, total=False):
    query: str
    medications: list[str]
    trials: list[TrialResult]
    interaction_warnings: list[InteractionWarning]
    answer: str


class BaseAgent(ABC):
    """Every agent node in the graph implements this. `name` is used for
    logging/tracing which agent produced which state change."""

    name: str = "base_agent"

    @abstractmethod
    async def run(self, state: AgentState) -> AgentState:
        ...

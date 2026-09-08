"""
FastAPI application entrypoint. Wires up all dependencies once at startup
(HTTP clients, vector store, agents, orchestrator) and stores them on
`app.state` so route handlers can access them without re-constructing
anything per-request.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.llm.ollama_client import OllamaClient
from app.retrieval.vector_store import TrialVectorStore
from app.clients.clinical_trials_client import ClinicalTrialsClient
from app.clients.drug_interaction_client import DrugInteractionClient
from app.agents.research_agent import ResearchAgent
from app.agents.interaction_agent import InteractionAgent
from app.agents.synthesis_agent import SynthesisAgent
from app.graph.orchestrator import TrialMatchOrchestrator
from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    ollama_client = OllamaClient()
    vector_store = TrialVectorStore(ollama_client=ollama_client)
    trials_client = ClinicalTrialsClient()
    interaction_client = DrugInteractionClient()

    research_agent = ResearchAgent(trials_client, vector_store)
    interaction_agent = InteractionAgent(interaction_client)
    synthesis_agent = SynthesisAgent(vector_store, ollama_client)

    app.state.ollama_client = ollama_client
    app.state.orchestrator = TrialMatchOrchestrator(research_agent, interaction_agent, synthesis_agent)

    yield  # app runs here

    # no explicit teardown needed - httpx clients are created per-call


app = FastAPI(
    title="TrialMatch Agents API",
    description="Multi-agent clinical trial intelligence system",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)

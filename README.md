# TrialMatch Agents

A multi-agent AI system that searches live clinical trial data, cross-references drug interactions, and synthesizes findings via orchestrated LLM agents — built on LangGraph, ChromaDB, and real-time ClinicalTrials.gov data.

## What this is

TrialMatch Agents takes a natural-language clinical query (e.g., "trials for stage 2 lung cancer patients also on warfarin and aspirin") and produces a synthesized, cited answer by coordinating three specialized agents through an explicit state graph — not a single long prompt pretending to be multiple steps.

This is a genuine step up from a typical single-agent RAG pipeline: each agent has a distinct responsibility, agents hand off state to one another, and the system deterministically guarantees safety-relevant information (drug interaction warnings) reaches the final answer, rather than leaving that entirely to the LLM's discretion.

## How it works

1. A **Research Agent** queries the real, public ClinicalTrials.gov API for trials matching the user's query, and indexes results into a vector store
2. An **Interaction Agent** cross-checks the patient's stated medications against a curated drug-interaction reference — both among the patient's own medications, and against any drugs mentioned in retrieved trial summaries
3. A **Synthesis Agent** retrieves the most relevant indexed trials via vector similarity search, and asks a local LLM (via Ollama) to write a coherent, cited narrative answer
4. Drug interaction warnings are **deterministically prepended** to the final answer — never solely dependent on the LLM choosing to mention them. This is a deliberate safety-critical design choice: instruction-following in an LLM is not a reliable enough guarantee for information this consequential
5. LangGraph orchestrates the hand-off between all three agents as an explicit state machine

## Core components

| Layer | What it does |
|---|---|
| `agents/` | `ResearchAgent`, `InteractionAgent`, `SynthesisAgent` — each implements a shared `BaseAgent` interface |
| `graph/` | LangGraph state machine (`orchestrator.py`) wiring the three agents together |
| `retrieval/` | ChromaDB vector store + Ollama-backed embedding function |
| `clients/` | Real ClinicalTrials.gov API wrapper, curated drug-interaction reference |
| `llm/` | Thin async wrapper around Ollama's generate/embed endpoints |
| `api/` | FastAPI app and routes (`/query`, `/health`) |
| `models/` | Pydantic request/response schemas shared across the app |

## Tech stack

- Python 3.11, FastAPI, Uvicorn
- LangGraph + LangChain Core (agent orchestration)
- ChromaDB (vector store)
- Ollama (local LLM — `llama3.1` for generation, `nomic-embed-text` for embeddings)
- ClinicalTrials.gov API v2 (real public data, no API key required)
- httpx, Pydantic
- pytest, pytest-asyncio, respx (HTTP mocking)
- Docker + docker-compose
- GitHub Actions CI

## Project structure

```text
trialmatch-agents/
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml
├── pytest.ini
├── app/
│ ├── agents/
│ │ ├── base_agent.py
│ │ ├── research_agent.py
│ │ ├── interaction_agent.py
│ │ └── synthesis_agent.py
│ ├── graph/
│ │ └── orchestrator.py
│ ├── retrieval/
│ │ ├── vector_store.py
│ │ └── embeddings.py
│ ├── clients/
│ │ ├── clinical_trials_client.py
│ │ └── drug_interaction_client.py
│ ├── llm/
│ │ └── ollama_client.py
│ ├── api/
│ │ ├── main.py
│ │ └── routes.py
│ └── models/
│ └── schemas.py
└── tests/
├── test_agents/
├── test_clients/
├── test_graph/
└── test_integration/
```


## Running it locally

```bash
# 1. Set up
cd trialmatch-agents
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# 2. Pull the Ollama models (locally, before Docker)
ollama pull llama3.1
ollama pull nomic-embed-text

# 3. Run the unit test suite (mocked — no live services needed)
pytest -v

# 4. Run the full stack (Ollama + app, containerized)
docker-compose up --build

# 5. Smoke test
curl http://localhost:8000/health
```

Note: on first `docker-compose up`, the containerized Ollama instance has its own separate model storage from any local Ollama install — pull the models inside the container too:

```bash
docker exec -it trialmatch-ollama ollama pull llama3.1
docker exec -it trialmatch-ollama ollama pull nomic-embed-text
```

## Verified working example

The full pipeline has been run end-to-end against real ClinicalTrials.gov data and a local Ollama instance:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "clinical trials for stage 2 lung cancer", "medications": ["warfarin", "aspirin"]}'
```

This returns real trial matches with genuine NCT IDs, a correctly flagged `warfarin` + `aspirin` MAJOR interaction warning (always present at the top of the response, deterministically), and an LLM-synthesized narrative that cites specific trials and reasons about relevance (e.g., correctly noting when a matched trial's population doesn't align with the query's stage).

## Testing

The unit test suite mocks all external calls (HTTP clients, LLM calls, vector store) and runs without any live services:

```bash
pytest -v
```

A separate real integration test exercises the full graph against the actual ClinicalTrials.gov API and a real local Ollama instance. It's opt-in and skipped by default (including in CI, which has no Ollama):

```bash
RUN_INTEGRATION_TESTS=1 pytest tests/test_integration -v
```

**Current results:** 13 unit tests passing, 1 integration test (skipped by default, passes when run manually against live services).

```bash
13 passed, 1 skipped
```


## Design notes worth calling out

- **Real data, not seeded/mocked data** — the Research Agent hits ClinicalTrials.gov's actual public API. There's no offline fallback dataset; this is a deliberate choice to keep the project honest about what "real" means.
- **The drug-interaction reference is a small curated dataset**, not a clinical-grade database — there's no free, comprehensive public API suitable for this project's scope. Worth being upfront about that distinction rather than overstating the system's clinical reliability.
- **Safety-relevant output is never purely LLM-dependent.** Interaction warnings are computed deterministically by `InteractionAgent` and structurally guaranteed to appear in the final response — the LLM's role is narrative framing, not the source of truth for whether a warning exists.
- **Ollama's embedding API has changed versions** (`/api/embeddings` legacy → `/api/embed` current, with different request/response shapes) — this project targets the current endpoint; worth checking your local Ollama version if embeddings ever start failing.

## License

MIT

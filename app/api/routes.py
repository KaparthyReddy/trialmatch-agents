from fastapi import APIRouter, Request

from app.models.schemas import QueryRequest, SynthesisResponse, HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    ollama_client = request.app.state.ollama_client
    reachable = await ollama_client.is_reachable()
    return HealthResponse(
        status="UP" if reachable else "DEGRADED",
        ollama_reachable=reachable,
        model=ollama_client.model,
    )


@router.post("/query", response_model=SynthesisResponse)
async def query(request: Request, body: QueryRequest) -> SynthesisResponse:
    orchestrator = request.app.state.orchestrator
    result = await orchestrator.run_query(body.query, body.medications)

    return SynthesisResponse(
        query=body.query,
        trials=result.get("trials", []),
        interaction_warnings=result.get("interaction_warnings", []),
        answer=result.get("answer", ""),
    )

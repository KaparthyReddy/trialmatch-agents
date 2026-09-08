"""
Pydantic request/response models shared across the API layer and agents.
Keeping these centralized means every agent and route speaks the same
shape, rather than passing around loose dicts.
"""

from typing import Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Natural-language clinical query")
    medications: list[str] = Field(default_factory=list, description="Medications the patient is currently on, if any")


class TrialResult(BaseModel):
    nct_id: str
    title: str
    status: str
    conditions: list[str]
    summary: str
    url: str


class InteractionWarning(BaseModel):
    drug_a: str
    drug_b: str
    severity: str
    note: str


class SynthesisResponse(BaseModel):
    query: str
    trials: list[TrialResult]
    interaction_warnings: list[InteractionWarning]
    answer: str


class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    model: str

"""
Wrapper around the real, public ClinicalTrials.gov API v2 (no API key
required). Docs: https://clinicaltrials.gov/data-api/api

This is genuinely live data — the ResearchAgent calls this to pull real
trial records matching a condition/term, not mocked or seeded data.
"""

import os
import httpx

from app.models.schemas import TrialResult


class ClinicalTrialsClient:

    def __init__(self, base_url: str | None = None, timeout: float = 15.0):
        self.base_url = (base_url or os.getenv(
            "CLINICALTRIALS_API_BASE", "https://clinicaltrials.gov/api/v2"
        )).rstrip("/")
        self.timeout = timeout

    async def search_trials(self, condition_or_term: str, max_results: int = 5) -> list[TrialResult]:
        params = {
            "query.term": condition_or_term,
            "pageSize": max_results,
            "format": "json",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/studies", params=params)
            response.raise_for_status()
            data = response.json()

        results = []
        for study in data.get("studies", []):
            protocol = study.get("protocolSection", {})
            identification = protocol.get("identificationModule", {})
            status_module = protocol.get("statusModule", {})
            conditions_module = protocol.get("conditionsModule", {})
            description_module = protocol.get("descriptionModule", {})

            nct_id = identification.get("nctId", "UNKNOWN")
            results.append(TrialResult(
                nct_id=nct_id,
                title=identification.get("briefTitle", "Untitled trial"),
                status=status_module.get("overallStatus", "UNKNOWN"),
                conditions=conditions_module.get("conditions", []),
                summary=description_module.get("briefSummary", "No summary available."),
                url=f"https://clinicaltrials.gov/study/{nct_id}",
            ))

        return results

import pytest
import httpx
import respx

from app.clients.clinical_trials_client import ClinicalTrialsClient

MOCK_RESPONSE = {
    "studies": [
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT00000001", "briefTitle": "Test Trial One"},
                "statusModule": {"overallStatus": "RECRUITING"},
                "conditionsModule": {"conditions": ["Lung Cancer"]},
                "descriptionModule": {"briefSummary": "A test trial summary."},
            }
        }
    ]
}


@pytest.mark.asyncio
@respx.mock
async def test_search_trials_parses_response():
    client = ClinicalTrialsClient(base_url="https://clinicaltrials.gov/api/v2")
    respx.get("https://clinicaltrials.gov/api/v2/studies").mock(
        return_value=httpx.Response(200, json=MOCK_RESPONSE)
    )

    results = await client.search_trials("lung cancer")

    assert len(results) == 1
    assert results[0].nct_id == "NCT00000001"
    assert results[0].status == "RECRUITING"
    assert "Lung Cancer" in results[0].conditions
    assert results[0].url == "https://clinicaltrials.gov/study/NCT00000001"


@pytest.mark.asyncio
@respx.mock
async def test_search_trials_handles_empty_results():
    client = ClinicalTrialsClient(base_url="https://clinicaltrials.gov/api/v2")
    respx.get("https://clinicaltrials.gov/api/v2/studies").mock(
        return_value=httpx.Response(200, json={"studies": []})
    )

    results = await client.search_trials("an extremely rare condition")
    assert results == []

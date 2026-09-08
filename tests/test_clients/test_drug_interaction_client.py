import pytest

from app.clients.drug_interaction_client import DrugInteractionClient


@pytest.fixture
def client():
    return DrugInteractionClient()


def test_finds_known_interaction(client):
    warnings = client.check_interactions(["warfarin", "aspirin"])
    assert len(warnings) == 1
    assert warnings[0].severity == "MAJOR"


def test_no_interaction_for_unrelated_drugs(client):
    warnings = client.check_interactions(["paracetamol", "vitamin c"])
    assert warnings == []


def test_case_and_whitespace_insensitive(client):
    warnings = client.check_interactions([" Warfarin ", "ASPIRIN"])
    assert len(warnings) == 1


def test_check_single_against_list(client):
    warnings = client.check_single_against_list("aspirin", ["warfarin"])
    assert len(warnings) == 1
    assert warnings[0].severity == "MAJOR"


def test_check_single_against_list_no_match(client):
    warnings = client.check_single_against_list("aspirin", ["metformin"])
    assert warnings == []

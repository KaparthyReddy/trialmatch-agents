"""
Lightweight local drug-interaction reference. There is no free, public,
comprehensive drug-interaction API suitable for this project's scope, so
this ships a small curated dataset of well-documented interactions
(similar in spirit to the seed data used in ColdChain Sentinel). This is
explicitly a reference lookup, not a clinical-grade interaction database —
worth being upfront about that distinction if this project comes up in
an interview.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class KnownInteraction:
    drug_a: str
    drug_b: str
    severity: str
    note: str


_KNOWN_INTERACTIONS: list[KnownInteraction] = [
    KnownInteraction("warfarin", "aspirin", "MAJOR",
                     "Combined anticoagulant/antiplatelet effect significantly increases bleeding risk"),
    KnownInteraction("metformin", "contrast dye", "MODERATE",
                     "Risk of lactic acidosis in patients with reduced renal function"),
    KnownInteraction("ssri", "maoi", "CONTRAINDICATED",
                     "Risk of serotonin syndrome; combination generally avoided"),
    KnownInteraction("simvastatin", "grapefruit juice", "MODERATE",
                     "Grapefruit inhibits CYP3A4 metabolism, raising statin plasma levels"),
    KnownInteraction("methotrexate", "nsaid", "MAJOR",
                     "NSAIDs can reduce methotrexate clearance, increasing toxicity risk"),
]


class DrugInteractionClient:

    def __init__(self):
        self._interactions = _KNOWN_INTERACTIONS

    def check_interactions(self, medications: list[str]) -> list[KnownInteraction]:
        normalized = [m.strip().lower() for m in medications]
        found = []

        for interaction in self._interactions:
            if interaction.drug_a in normalized and interaction.drug_b in normalized:
                found.append(interaction)

        return found

    def check_single_against_list(self, drug: str, medications: list[str]) -> list[KnownInteraction]:
        """Checks one drug (e.g. one mentioned in a trial) against a patient's existing medication list."""
        drug_normalized = drug.strip().lower()
        normalized_meds = [m.strip().lower() for m in medications]
        found = []

        for interaction in self._interactions:
            pair = {interaction.drug_a, interaction.drug_b}
            if drug_normalized in pair and any(med in pair for med in normalized_meds):
                found.append(interaction)

        return found

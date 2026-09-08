"""
Cross-checks the patient's stated medications against a curated
drug-interaction reference, and separately flags any interaction risk
between a patient's medication and a drug mentioned in a retrieved trial's
summary. This is the agent that turns raw trial matches into something
clinically cautious rather than naively presented.
"""

from app.agents.base_agent import BaseAgent, AgentState
from app.clients.drug_interaction_client import DrugInteractionClient
from app.models.schemas import InteractionWarning


class InteractionAgent(BaseAgent):
    name = "interaction_agent"

    def __init__(self, interaction_client: DrugInteractionClient):
        self.interaction_client = interaction_client

    async def run(self, state: AgentState) -> AgentState:
        medications = state.get("medications", [])
        warnings: list[InteractionWarning] = []

        # Interactions among the patient's own medication list
        for interaction in self.interaction_client.check_interactions(medications):
            warnings.append(InteractionWarning(
                drug_a=interaction.drug_a,
                drug_b=interaction.drug_b,
                severity=interaction.severity,
                note=interaction.note,
            ))

        # Interactions between a patient's medication and a drug mentioned
        # in a retrieved trial's summary text
        for trial in state.get("trials", []):
            summary_lower = trial.summary.lower()
            for medication in medications:
                mentioned_drugs = self._extract_candidate_drug_mentions(summary_lower)
                for candidate in mentioned_drugs:
                    for interaction in self.interaction_client.check_single_against_list(candidate, [medication]):
                        warnings.append(InteractionWarning(
                            drug_a=interaction.drug_a,
                            drug_b=interaction.drug_b,
                            severity=interaction.severity,
                            note=f"{interaction.note} (flagged from trial {trial.nct_id})",
                        ))

        state["interaction_warnings"] = warnings
        return state

    @staticmethod
    def _extract_candidate_drug_mentions(text: str) -> list[str]:
        """Very deliberately simple: checks trial summary text for any drug
        name that appears in the known-interaction reference set, rather
        than attempting real NLP entity extraction (out of scope here)."""
        known_drug_names = {
            "warfarin", "aspirin", "metformin", "ssri", "maoi",
            "simvastatin", "methotrexate", "nsaid",
        }
        return [name for name in known_drug_names if name in text]

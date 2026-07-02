"""Eval task definitions.

A task owns three things: the prompt it puts to the agent, the ground truth it
derives from the case data, and how the agent's raw output is parsed into
comparable values. Adding a task type means adding a class with these three
methods — the runner and scorers don't change.
"""

import json
import logging
import re
from dataclasses import dataclass

from healthchain.fhir import get_resources

from groundeval.datasets import EvalCase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Medication:
    """A single extracted or ground-truth medication."""

    rxnorm_code: str
    name: str


class MedicationExtractionTask:
    """Extract a patient's active medications from their FHIR bundle.

    Ground truth is derived from the bundle itself: every MedicationRequest
    with status "active", identified by its RxNorm code. The agent sees the
    same bundle as JSON and must return the active medications — so the task
    measures whether the model can read clinical data accurately, not whether
    it knows medicine.
    """

    name = "medication-extraction"

    def prompt(self, case: EvalCase) -> str:
        return (
            "You are given a FHIR R4 bundle for a single patient.\n"
            "Identify the patient's ACTIVE medications: every distinct medication "
            "in a MedicationRequest whose status is exactly \"active\". Exclude "
            "completed or stopped medications. List each distinct medication once.\n\n"
            "Respond with ONLY a JSON object in this exact shape, no other text:\n"
            '{"active_medications": [{"name": "<display name>", "rxnorm_code": "<RxNorm code>"}]}\n'
            'If there are no active medications, return {"active_medications": []}.\n\n'
            f"FHIR bundle:\n{case.bundle_json}"
        )

    def ground_truth(self, case: EvalCase) -> set[Medication]:
        meds = set()
        for req in get_resources(case.bundle, "MedicationRequest"):
            if req.status != "active":
                continue
            concept = req.medicationCodeableConcept
            if concept is None or not concept.coding:
                # Active requests in the fixture set always carry an inline
                # RxNorm coding; a reference-only active request would need
                # resolution we deliberately don't do in v0.1.
                logger.warning(
                    "Case %s: active MedicationRequest without inline coding skipped",
                    case.case_id,
                )
                continue
            coding = concept.coding[0]
            meds.add(Medication(rxnorm_code=str(coding.code), name=str(coding.display or "")))
        return meds

    def parse_output(self, raw: str) -> set[Medication]:
        """Parse the agent's JSON reply into Medication values.

        Tolerates code fences and surrounding prose; raises ValueError if no
        parseable JSON object is found so the runner can record a failed case
        rather than a silently empty prediction.
        """
        text = raw.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object in agent output: {text[:200]!r}")
        payload = json.loads(match.group(0))
        return {
            Medication(rxnorm_code=str(m.get("rxnorm_code", "")), name=str(m.get("name", "")))
            for m in payload.get("active_medications", [])
        }

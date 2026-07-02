"""Eval task definitions.

A task owns everything about its values: the prompt it puts to the agent, the
ground truth it derives from the case data, how the agent's raw output is
parsed into comparable values, and how those values are shaped for recording
(`empty_output`, `record`). Adding a task type means adding a class with these
methods — the runner stays value-agnostic and doesn't change.
"""

import json
import logging
import re
from dataclasses import dataclass

from healthchain.fhir import get_resources

from groundeval.datasets import EvalCase, WriteCase

logger = logging.getLogger(__name__)


def _json_object(raw: str) -> dict:
    """Extract the JSON object from agent output, tolerating fences and prose."""
    match = re.search(r"\{.*\}", raw.strip(), re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object in agent output: {raw.strip()[:200]!r}")
    return json.loads(match.group(0))


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

    def empty_output(self) -> set[Medication]:
        """What "no prediction" looks like, e.g. when the agent call fails."""
        return set()

    def record(self, values: set[Medication]) -> list[Medication]:
        """Stable ordered form of task values for run records and traces."""
        return sorted(values, key=lambda m: m.rxnorm_code)

    def parse_output(self, raw: str) -> set[Medication]:
        """Parse the agent's JSON reply into Medication values.

        Tolerates code fences and surrounding prose; raises ValueError if no
        parseable JSON object is found so the runner can record a failed case
        rather than a silently empty prediction.
        """
        payload = _json_object(raw)
        return {
            Medication(rxnorm_code=str(m.get("rxnorm_code", "")), name=str(m.get("name", "")))
            for m in payload.get("active_medications", [])
        }


@dataclass(frozen=True)
class DosageSpec:
    """The regimen the source text documents: dose per administration."""

    dose_value: float
    dose_unit: str
    frequency_per_day: float


@dataclass(frozen=True)
class WriteExpectation:
    """Ground truth for one write case.

    `max_daily_dose` (in `dose_unit`) is a per-drug safety ceiling, not part
    of the narrative — it exists so the scorer can catch unit and arithmetic
    errors (81 g instead of 81 mg) that are perfectly valid FHIR.
    """

    subject: str
    rxnorm_code: str
    display: str
    status: str
    dosage: DosageSpec | None
    max_daily_dose: float | None


class WriteAndValidateTask:
    """Generate a FHIR MedicationStatement from a clinical note excerpt.

    The inverse of extraction: instead of reading structured data, the agent
    must *write* it — find the right RxNorm code (via lookup, not memory),
    pick a status that reflects the narrative, and encode the documented
    regimen as structured dosage. Ground truth is the committed expectation
    block for each case; scoring is deterministic against it.

    The prompt states the site's write policy because an agent can only be
    held to rules it was given. It never contains expected values.
    """

    name = "write-and-validate"

    def prompt(self, case: WriteCase) -> str:
        return (
            "You are a clinical data engineer agent writing to a hospital EHR.\n"
            "Given a clinical note excerpt about ONE medication, produce the FHIR R4B "
            "MedicationStatement resource that documents it.\n\n"
            f"Patient: {case.subject}\n"
            f"Clinical note: {case.input_text}\n\n"
            "Site write policy (every write is checked against this):\n"
            "- The medication must be coded in RxNorm. Find the code with the lookup "
            "tool using the site catalog — never guess or recall a code. Use the "
            "catalog's code and display name exactly.\n"
            f"- The resource must reference exactly this patient: {case.subject}.\n"
            "- status must be a valid R4B MedicationStatement status and accurately "
            "reflect the note (e.g. currently taking vs. finished vs. stopped vs. denies "
            "taking).\n"
            "- If the note documents how the medication is or was taken, include "
            "structured dosage: the dose actually taken per single administration, its "
            "unit, and administrations per day. If the note documents no regimen, do "
            "not invent one.\n\n"
            "Build the resource with the tools and validate it before answering.\n"
            "Final answer: ONLY the MedicationStatement resource JSON, no other text."
        )

    def ground_truth(self, case: WriteCase) -> WriteExpectation:
        e = case.expected
        dosage = DosageSpec(**e["dosage"]) if e["dosage"] else None
        return WriteExpectation(
            subject=case.subject,
            rxnorm_code=e["rxnorm_code"],
            display=e["display"],
            status=e["status"],
            dosage=dosage,
            max_daily_dose=e["max_daily_dose"],
        )

    def empty_output(self) -> dict:
        """What "no prediction" looks like, e.g. when the agent call fails."""
        return {}

    def record(self, values) -> object:
        """Both value shapes (expectation dataclass, resource dict) are
        JSON-serializable as-is by the tracing layer."""
        return values

    def parse_output(self, raw: str) -> dict:
        """The agent's final answer is the generated resource itself."""
        resource = _json_object(raw)
        if not isinstance(resource, dict):
            raise ValueError(f"Agent output is not a JSON object: {raw[:200]!r}")
        return resource

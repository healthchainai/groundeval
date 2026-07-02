"""HealthChain FHIR operations exposed as agent tools.

This module is the adapter layer between HealthChain's FHIR helpers and a
tool-calling agent, and nothing else: no eval logic, no scoring, no knowledge
of tasks. It is written to be lifted into HealthChain wholesale (README
roadmap item 5), which drives three conventions:

- **Flat scalar arguments.** Models emit flat JSON arguments far more reliably
  than nested FHIR structures, so each tool takes strings/numbers and the
  adapter does the FHIR nesting (see `build_medication_statement`, which
  hand-rolls the R4B Dosage tree HealthChain's helper can't produce).
- **Errors are return values, not exceptions.** A tool that raises kills the
  agent loop; a tool that returns `{"ok": false, "errors": [...]}` gives the
  model something to read and correct. Every tool returns a JSON string.
- **Validation reports what the spec requires, not just what Pydantic
  checks.** `fhir.resources` does not enforce required ValueSet bindings
  (e.g. MedicationStatement.status), so `_validate` layers those checks on
  top of model validation.

The terminology lookup is backed by a committed table derived from the
Synthea fixtures (see scripts/make_terminology.py) — an offline stand-in for
a site terminology service, so agents look codes up instead of guessing them
from parametric memory.
"""

import json
from pathlib import Path

from fhir.resources.R4B.dosage import Dosage, DosageDoseAndRate
from fhir.resources.R4B.medicationstatement import MedicationStatement
from fhir.resources.R4B.quantity import Quantity
from fhir.resources.R4B.timing import Timing, TimingRepeat
from healthchain.fhir import create_medication_statement

RXNORM_SYSTEM = "http://www.nlm.nih.gov/research/umls/rxnorm"

# The R4B required binding for MedicationStatement.status. fhir.resources
# records these in json_schema_extra but does not enforce them, and
# HealthChain's create_medication_statement defaults to "recorded" — an R5
# status that is invalid here — so the adapter must own this check.
R4B_MEDICATION_STATEMENT_STATUSES = frozenset(
    {"active", "completed", "entered-in-error", "intended",
     "stopped", "on-hold", "unknown", "not-taken"}
)

DEFAULT_TERMINOLOGY_PATH = (
    Path(__file__).parent.parent.parent / "data" / "terminology" / "rxnorm_medications.json"
)

_terminology: list[dict] | None = None
_terminology_path = DEFAULT_TERMINOLOGY_PATH


def configure_terminology(path: Path | str) -> None:
    """Point the lookup tool at a different terminology table (e.g. in tests)."""
    global _terminology, _terminology_path
    _terminology_path = Path(path)
    _terminology = None


def _table() -> list[dict]:
    global _terminology
    if _terminology is None:
        _terminology = json.loads(_terminology_path.read_text())
    return _terminology


def lookup_medication_code(query: str) -> str:
    """Search the site medication catalog for RxNorm codings by name.

    Matches medications whose display name contains every word in the query
    (case-insensitive), so brand names, ingredient names, and strengths all
    work: "tylenol", "metoprolol succinate 25", "aspirin 81". Returns a JSON
    object with a "matches" list of {code, display, system}. Use a code from
    the matches — never invent one. If there are several matches, refine the
    query with the strength or form until it is unambiguous.
    """
    tokens = query.lower().split()
    matches = [
        entry
        for entry in _table()
        if all(
            any(tok in word for word in entry["display"].lower().split())
            for tok in tokens
        )
    ]
    return json.dumps({"matches": matches, "count": len(matches)})


def _validate(resource_dict: dict) -> list[str]:
    """Validate a MedicationStatement dict; return issues (empty = valid).

    Two layers: Pydantic model validation for structure, then the required
    ValueSet bindings Pydantic skips.
    """
    issues: list[str] = []
    if resource_dict.get("resourceType") != "MedicationStatement":
        issues.append(
            f"resourceType must be 'MedicationStatement', got {resource_dict.get('resourceType')!r}"
        )
        return issues
    try:
        MedicationStatement.model_validate(resource_dict)
    except Exception as e:
        # Pydantic's multi-error reports are verbose; one line per error keeps
        # them readable for a model.
        issues.extend(line.strip() for line in str(e).splitlines() if line.strip())
        return issues

    status = resource_dict.get("status")
    if status not in R4B_MEDICATION_STATEMENT_STATUSES:
        issues.append(
            f"status {status!r} is not a valid R4B MedicationStatement status; "
            f"allowed: {', '.join(sorted(R4B_MEDICATION_STATEMENT_STATUSES))}"
        )
    concept = resource_dict.get("medicationCodeableConcept")
    if not (concept and concept.get("coding")):
        issues.append("medicationCodeableConcept must carry at least one coding")
    return issues


def build_medication_statement(
    subject: str,
    status: str,
    rxnorm_code: str,
    display: str,
    dose_value: float | None = None,
    dose_unit: str | None = None,
    frequency_per_day: int | None = None,
) -> str:
    """Build and validate a FHIR R4B MedicationStatement.

    Args:
        subject: Patient reference, e.g. "Patient/pat-001".
        status: R4B MedicationStatement status reflecting the source text
            (active, completed, stopped, not-taken, intended, on-hold,
            unknown, entered-in-error).
        rxnorm_code: RxNorm code from lookup_medication_code — never guessed.
        display: The display name that accompanies the code in the catalog.
        dose_value: Amount taken per single administration (e.g. 400 for
            "two 200 mg tablets"). Omit if the source documents no regimen.
        dose_unit: Unit for dose_value, e.g. "mg". Required with dose_value.
        frequency_per_day: Administrations per day (e.g. 4 for "every 6
            hours"). Required with dose_value.

    Returns a JSON object: {"ok": true, "resource": {...}} on success, or
    {"ok": false, "errors": [...]} describing what to fix.
    """
    dosage_args = (dose_value, dose_unit, frequency_per_day)
    if any(a is not None for a in dosage_args) and not all(a is not None for a in dosage_args):
        return json.dumps({
            "ok": False,
            "errors": ["dose_value, dose_unit, and frequency_per_day must be provided together"],
        })

    try:
        ms = create_medication_statement(
            subject=subject,
            status=status,
            code=rxnorm_code,
            display=display,
            system=RXNORM_SYSTEM,
        )
        if dose_value is not None:
            ms.dosage = [
                Dosage(
                    text=f"{dose_value:g} {dose_unit}, {frequency_per_day}x per day",
                    timing=Timing(
                        repeat=TimingRepeat(frequency=frequency_per_day, period=1, periodUnit="d")
                    ),
                    doseAndRate=[
                        DosageDoseAndRate(doseQuantity=Quantity(value=dose_value, unit=dose_unit))
                    ],
                )
            ]
    except Exception as e:
        return json.dumps({"ok": False, "errors": [f"{type(e).__name__}: {e}"]})

    resource = json.loads(ms.model_dump_json(exclude_none=True))
    issues = _validate(resource)
    if issues:
        return json.dumps({"ok": False, "errors": issues})
    return json.dumps({"ok": True, "resource": resource})


def validate_medication_statement(resource_json: str) -> str:
    """Validate a MedicationStatement JSON string against FHIR R4B.

    Checks Pydantic model validity plus the required status binding that
    schema validation alone misses. Returns {"valid": true} or
    {"valid": false, "issues": [...]}.
    """
    try:
        resource = json.loads(resource_json)
    except json.JSONDecodeError as e:
        return json.dumps({"valid": False, "issues": [f"not valid JSON: {e}"]})
    if not isinstance(resource, dict):
        return json.dumps({"valid": False, "issues": ["expected a JSON object"]})
    issues = _validate(resource)
    return json.dumps({"valid": not issues, "issues": issues})


def get_langchain_tools() -> list:
    """The adapters wrapped as LangChain tools for a LangGraph agent.

    Import is local so the plain functions stay usable without LangChain
    installed — the module's only hard dependencies are HealthChain and
    fhir.resources.
    """
    from langchain_core.tools import tool

    return [
        tool(lookup_medication_code),
        tool(build_medication_statement),
        tool(validate_medication_statement),
    ]

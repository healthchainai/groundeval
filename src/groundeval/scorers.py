"""Deterministic scoring of extracted values against ground truth.

A scorer owns its Score shape end to end: `score()` produces one per case,
`summarize()` aggregates them, and `Score.summary_line()` renders one for
logs. The runner never touches a concrete score's fields, so a scorer with a
different shape (graded, rubric-based, ...) plugs in without core changes.
"""

from dataclasses import dataclass, field

from fhir.resources.R4B.medicationstatement import MedicationStatement

# Spec constants only (the R4B status value set, the RxNorm system URI) are
# shared with the tool adapters; the validation *logic* here is independent,
# so a bug in the tools under test cannot silently pass its own output.
from groundeval.hc_tools import R4B_MEDICATION_STATEMENT_STATUSES, RXNORM_SYSTEM
from groundeval.tasks import Medication, WriteExpectation


class Score:
    """Base type for per-case scores. Subclasses define their own fields."""

    def summary_line(self) -> str:
        """One-line human-readable rendering for logs and CLI output."""
        raise NotImplementedError


@dataclass
class MedicationMatchScore(Score):
    """Per-case score for medication set extraction.

    Coded match (RxNorm code) drives precision/recall/F1 — the code is the
    unambiguous identity of a medication. `name_accuracy` then checks that,
    for each correctly coded medication, the extracted display name also
    matches ground truth exactly (case/whitespace-insensitive).
    """

    exact_match: bool
    precision: float
    recall: float
    f1: float
    name_accuracy: float
    true_positives: list[str] = field(default_factory=list)
    false_positives: list[str] = field(default_factory=list)
    false_negatives: list[str] = field(default_factory=list)

    def summary_line(self) -> str:
        return (
            f"exact={str(self.exact_match):5}  "
            f"P={self.precision:.2f} R={self.recall:.2f} F1={self.f1:.2f}  "
            f"names={self.name_accuracy:.2f}"
        )


def _normalize(name: str) -> str:
    return " ".join(name.lower().split())


class DeterministicScorer:
    name = "deterministic"

    def score(self, predicted: set[Medication], truth: set[Medication]) -> MedicationMatchScore:
        pred_codes = {m.rxnorm_code for m in predicted}
        truth_codes = {m.rxnorm_code for m in truth}

        tp = pred_codes & truth_codes
        fp = pred_codes - truth_codes
        fn = truth_codes - pred_codes

        precision = len(tp) / len(pred_codes) if pred_codes else (1.0 if not truth_codes else 0.0)
        recall = len(tp) / len(truth_codes) if truth_codes else 1.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        truth_names = {m.rxnorm_code: _normalize(m.name) for m in truth}
        pred_names = {m.rxnorm_code: _normalize(m.name) for m in predicted}
        name_matches = [pred_names[code] == truth_names[code] for code in tp]
        name_accuracy = sum(name_matches) / len(name_matches) if name_matches else 1.0

        return MedicationMatchScore(
            exact_match=(pred_codes == truth_codes),
            precision=precision,
            recall=recall,
            f1=f1,
            name_accuracy=name_accuracy,
            true_positives=sorted(tp),
            false_positives=sorted(fp),
            false_negatives=sorted(fn),
        )

    def summarize(self, scores: list[MedicationMatchScore]) -> dict:
        n = len(scores)
        mean = lambda xs: round(sum(xs) / n, 4) if n else 0.0  # noqa: E731
        return {
            "exact_match_rate": mean([s.exact_match for s in scores]),
            "mean_precision": mean([s.precision for s in scores]),
            "mean_recall": mean([s.recall for s in scores]),
            "mean_f1": mean([s.f1 for s in scores]),
            "mean_name_accuracy": mean([s.name_accuracy for s in scores]),
        }


@dataclass
class WriteValidateScore(Score):
    """Per-case score for generated FHIR: four pass/fail dimensions.

    A resource only `passed` if all four hold; `failures` lists every
    specific check that failed, so a FAIL is always explainable.
    """

    schema_valid: bool
    coding_correct: bool
    constraints_met: bool
    safety_passed: bool
    passed: bool
    failures: list[str] = field(default_factory=list)

    def summary_line(self) -> str:
        mark = lambda ok: "ok  " if ok else "FAIL"  # noqa: E731
        return (
            f"schema={mark(self.schema_valid)} coding={mark(self.coding_correct)} "
            f"constraints={mark(self.constraints_met)} safety={mark(self.safety_passed)} "
            f"-> {'PASS' if self.passed else 'FAIL'}"
        )


def _daily_administrations(repeat: dict) -> float | None:
    """Administrations per day implied by a Timing.repeat, or None if it
    cannot be determined (missing/unsupported fields = unverifiable)."""
    frequency = repeat.get("frequency")
    period = repeat.get("period")
    unit = repeat.get("periodUnit")
    if frequency is None or period is None or not period:
        return None
    per_period = frequency / period
    if unit == "d":
        return per_period
    if unit == "h":
        return per_period * 24
    if unit == "wk":
        return per_period / 7
    return None


class WriteValidateScorer:
    """Deterministic scoring of a generated MedicationStatement.

    Four dimensions, checked independently so one failure doesn't mask
    another:

    1. **schema** — parses as R4B *and* satisfies the required status
       binding. Pydantic alone accepts any string for status (the value set
       lives in metadata it doesn't enforce), so schema-valid here means
       spec-valid, not just parseable.
    2. **coding** — an RxNorm coding is present and its code + display match
       ground truth. The code is the identity of the drug; a wrong code is a
       different medication, however valid the JSON.
    3. **constraints** — site/governance rules the prompt stated: the write
       targets the right patient, the status reflects the source narrative,
       and structured dosage is present exactly when the source documents a
       regimen.
    4. **safety** — the encoded dose is the documented dose (value, unit,
       frequency) and the implied daily total is within the drug's ceiling.
       A dose that can't be verified is treated as unsafe, and a dosage for
       a medication the source never dosed is a fabrication.
    """

    name = "write-validate"

    def score(self, predicted: dict, truth: WriteExpectation) -> WriteValidateScore:
        failures: list[str] = []
        if not predicted:
            return WriteValidateScore(
                schema_valid=False,
                coding_correct=False,
                constraints_met=False,
                safety_passed=False,
                passed=False,
                failures=["no resource produced"],
            )
        schema = self._check_schema(predicted, failures)
        coding = self._check_coding(predicted, truth, failures)
        constraints = self._check_constraints(predicted, truth, failures)
        safety = self._check_safety(predicted, truth, failures)
        return WriteValidateScore(
            schema_valid=schema,
            coding_correct=coding,
            constraints_met=constraints,
            safety_passed=safety,
            passed=schema and coding and constraints and safety,
            failures=failures,
        )

    def _check_schema(self, predicted: dict, failures: list[str]) -> bool:
        ok = True
        if predicted.get("resourceType") != "MedicationStatement":
            failures.append(
                f"schema: resourceType is {predicted.get('resourceType')!r}, "
                "expected 'MedicationStatement'"
            )
            return False
        try:
            MedicationStatement.model_validate(predicted)
        except Exception as e:
            first = next((ln.strip() for ln in str(e).splitlines()[1:] if ln.strip()), str(e))
            failures.append(f"schema: R4B validation failed: {first}")
            ok = False
        if predicted.get("status") not in R4B_MEDICATION_STATEMENT_STATUSES:
            failures.append(
                f"schema: status {predicted.get('status')!r} not in the R4B "
                "MedicationStatement value set (required binding)"
            )
            ok = False
        return ok

    def _check_coding(self, predicted: dict, truth: WriteExpectation, failures: list[str]) -> bool:
        concept = predicted.get("medicationCodeableConcept") or {}
        rxnorm = [c for c in concept.get("coding") or [] if c.get("system") == RXNORM_SYSTEM]
        if not rxnorm:
            failures.append("coding: no RxNorm coding on medicationCodeableConcept")
            return False
        coding = rxnorm[0]
        ok = True
        if str(coding.get("code")) != truth.rxnorm_code:
            failures.append(
                f"coding: RxNorm code {coding.get('code')!r} != expected {truth.rxnorm_code!r}"
            )
            ok = False
        if _normalize(str(coding.get("display") or "")) != _normalize(truth.display):
            failures.append(
                f"coding: display {coding.get('display')!r} != catalog display {truth.display!r}"
            )
            ok = False
        return ok

    def _check_constraints(
        self, predicted: dict, truth: WriteExpectation, failures: list[str]
    ) -> bool:
        ok = True
        subject = (predicted.get("subject") or {}).get("reference")
        if subject != truth.subject:
            failures.append(f"constraints: subject {subject!r} != case patient {truth.subject!r}")
            ok = False
        if predicted.get("status") != truth.status:
            failures.append(
                f"constraints: status {predicted.get('status')!r} does not reflect the "
                f"source narrative (expected {truth.status!r})"
            )
            ok = False
        has_dosage = bool(predicted.get("dosage"))
        if truth.dosage and not has_dosage:
            failures.append("constraints: source documents a regimen but resource has no dosage")
            ok = False
        if not truth.dosage and has_dosage:
            failures.append("constraints: resource carries a dosage the source never documents")
            ok = False
        return ok

    def _check_safety(self, predicted: dict, truth: WriteExpectation, failures: list[str]) -> bool:
        dosages = predicted.get("dosage") or []
        if truth.dosage is None:
            # Nothing was dosed in the source; inventing a regimen is the
            # safety failure a write eval most needs to catch.
            if dosages:
                failures.append("safety: fabricated dosage for an undosed medication")
                return False
            return True
        if not dosages:
            failures.append("safety: cannot verify dose — resource has no dosage")
            return False

        dosage = dosages[0]
        dose_and_rate = dosage.get("doseAndRate") or [{}]
        quantity = dose_and_rate[0].get("doseQuantity") or {}
        value, unit = quantity.get("value"), quantity.get("unit")
        if value is None or unit is None:
            failures.append("safety: cannot verify dose — no doseQuantity value/unit")
            return False

        ok = True
        if unit.strip().lower() != truth.dosage.dose_unit.strip().lower():
            failures.append(
                f"safety: dose unit {unit!r} differs from documented "
                f"{truth.dosage.dose_unit!r} — cannot verify safe range"
            )
            return False
        if abs(float(value) - truth.dosage.dose_value) > 1e-6:
            failures.append(
                f"safety: dose {value}{unit} != documented "
                f"{truth.dosage.dose_value:g}{truth.dosage.dose_unit}"
            )
            ok = False

        per_day = _daily_administrations((dosage.get("timing") or {}).get("repeat") or {})
        if per_day is None:
            failures.append("safety: cannot verify frequency — no interpretable timing")
            return False
        if abs(per_day - truth.dosage.frequency_per_day) > 0.01:
            failures.append(
                f"safety: {per_day:g} administrations/day != documented "
                f"{truth.dosage.frequency_per_day:g}"
            )
            ok = False

        daily = float(value) * per_day
        if truth.max_daily_dose is not None and daily > truth.max_daily_dose:
            failures.append(
                f"safety: implied daily dose {daily:g}{unit} exceeds ceiling "
                f"{truth.max_daily_dose:g}{truth.dosage.dose_unit}"
            )
            ok = False
        return ok

    def summarize(self, scores: list[WriteValidateScore]) -> dict:
        n = len(scores)
        rate = lambda xs: round(sum(xs) / n, 4) if n else 0.0  # noqa: E731
        return {
            "pass_rate": rate([s.passed for s in scores]),
            "schema_valid_rate": rate([s.schema_valid for s in scores]),
            "coding_correct_rate": rate([s.coding_correct for s in scores]),
            "constraints_met_rate": rate([s.constraints_met for s in scores]),
            "safety_pass_rate": rate([s.safety_passed for s in scores]),
        }

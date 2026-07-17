"""Tests for the write-and-validate slice.

Offline coverage: fixture integrity, each tool adapter (including its error
paths — a tool that raises instead of returning errors would kill the agent
loop), each scorer dimension failing independently, and the full pipeline via
stub agents. `test_live_slice` runs the real LangGraph agent and skips when
no ANTHROPIC_API_KEY is set.
"""

import json
import os

import pytest

from groundeval import hc_tools, runner
from groundeval.datasets import load_write_cases
from groundeval.scorers import WriteValidateScorer
from groundeval.tasks import WriteAndValidateTask
from groundeval.tracing import init_tracing


@pytest.fixture(scope="module")
def cases():
    return load_write_cases()


@pytest.fixture(scope="module")
def terminology():
    return {e["code"]: e for e in json.loads(hc_tools.DEFAULT_TERMINOLOGY_PATH.read_text())}


def build_expected(case) -> dict:
    """The correct resource for a case, built through the tool adapters —
    which doubles as proof that tool output is pass-grade for the scorer."""
    e = case.expected
    dosage = e["dosage"] or {}
    reply = json.loads(
        hc_tools.build_medication_statement(
            subject=case.subject,
            status=e["status"],
            rxnorm_code=e["rxnorm_code"],
            display=e["display"],
            dose_value=dosage.get("dose_value"),
            dose_unit=dosage.get("dose_unit"),
            frequency_per_day=dosage.get("frequency_per_day"),
        )
    )
    assert reply["ok"], reply
    return reply["resource"]


def test_fixture_integrity(cases, terminology):
    assert len(cases) >= 5
    statuses = set()
    for case in cases:
        e = case.expected
        # ground-truth codings must exist in the committed catalog, verbatim
        assert e["rxnorm_code"] in terminology, case.case_id
        assert terminology[e["rxnorm_code"]]["display"] == e["display"], case.case_id
        statuses.add(e["status"])
        if e["dosage"]:
            # the documented regimen must itself sit inside the safety ceiling
            daily = e["dosage"]["dose_value"] * e["dosage"]["frequency_per_day"]
            assert daily <= e["max_daily_dose"], case.case_id
    # the set exercises more than the happy "active" path
    assert {"active", "completed", "stopped", "not-taken"} <= statuses


def test_lookup_tool():
    hits = json.loads(hc_tools.lookup_medication_code("tylenol"))
    assert hits["count"] == 1 and hits["matches"][0]["code"] == "209387"

    ambiguous = json.loads(hc_tools.lookup_medication_code("metoprolol succinate"))
    assert ambiguous["count"] == 2
    refined = json.loads(hc_tools.lookup_medication_code("metoprolol succinate 25"))
    assert refined["count"] == 1 and refined["matches"][0]["code"] == "866427"

    assert json.loads(hc_tools.lookup_medication_code("warfarin"))["count"] == 0


def test_build_tool_happy_path():
    reply = json.loads(
        hc_tools.build_medication_statement(
            "Patient/pat-001", "active", "243670", "aspirin 81 MG Oral Tablet", 81, "mg", 1
        )
    )
    assert reply["ok"]
    resource = reply["resource"]
    assert resource["status"] == "active"
    assert resource["subject"]["reference"] == "Patient/pat-001"
    assert resource["dosage"][0]["doseAndRate"][0]["doseQuantity"] == {"value": 81, "unit": "mg"}
    assert resource["dosage"][0]["timing"]["repeat"]["frequency"] == 1


def test_build_tool_errors_are_values():
    # "recorded" is the R5 status HealthChain itself defaults to; the adapter
    # must reject it for R4B with a message the agent can act on
    reply = json.loads(
        hc_tools.build_medication_statement("Patient/p", "recorded", "243670", "aspirin")
    )
    assert not reply["ok"]
    assert "R4B" in reply["errors"][0] and "active" in reply["errors"][0]

    partial = json.loads(
        hc_tools.build_medication_statement(
            "Patient/p", "active", "243670", "aspirin", dose_value=81
        )
    )
    assert not partial["ok"] and "together" in partial["errors"][0]


def test_validate_tool():
    assert not json.loads(hc_tools.validate_medication_statement("not json"))["valid"]

    r5_status = {
        "resourceType": "MedicationStatement",
        "status": "recorded",
        "subject": {"reference": "Patient/p"},
        "medicationCodeableConcept": {"coding": [{"code": "1", "system": "x"}]},
    }
    verdict = json.loads(hc_tools.validate_medication_statement(json.dumps(r5_status)))
    assert not verdict["valid"] and "R4B" in verdict["issues"][0]

    good = json.loads(
        hc_tools.build_medication_statement("Patient/p", "active", "243670", "aspirin")
    )["resource"]
    assert json.loads(hc_tools.validate_medication_statement(json.dumps(good)))["valid"]


class TestScorerDimensions:
    """Each dimension must be able to fail on its own, with a named failure."""

    @pytest.fixture()
    def scorer(self):
        return WriteValidateScorer()

    @pytest.fixture()
    def aspirin(self, cases):
        return next(c for c in cases if c.case_id == "aspirin-active-daily")

    @pytest.fixture()
    def truth(self, aspirin):
        return WriteAndValidateTask().ground_truth(aspirin)

    @pytest.fixture()
    def resource(self, aspirin):
        return build_expected(aspirin)

    def test_perfect_resource_passes(self, scorer, truth, resource):
        s = scorer.score(resource, truth)
        assert s.passed and not s.failures, s.failures

    def test_empty_prediction_fails_everything(self, scorer, truth):
        s = scorer.score({}, truth)
        assert not (s.schema_valid or s.coding_correct or s.constraints_met or s.safety_passed)

    def test_wrong_code_fails_coding_only(self, scorer, truth, resource):
        resource["medicationCodeableConcept"]["coding"][0]["code"] = "314076"
        s = scorer.score(resource, truth)
        assert not s.coding_correct
        assert s.schema_valid and s.constraints_met and s.safety_passed
        assert any("coding:" in f for f in s.failures)

    def test_wrong_patient_fails_constraints_only(self, scorer, truth, resource):
        resource["subject"]["reference"] = "Patient/pat-999"
        s = scorer.score(resource, truth)
        assert not s.constraints_met
        assert s.schema_valid and s.coding_correct and s.safety_passed

    def test_r5_status_fails_schema(self, scorer, truth, resource):
        resource["status"] = "recorded"
        s = scorer.score(resource, truth)
        assert not s.schema_valid  # pydantic alone would accept this
        assert not s.constraints_met  # and it no longer reflects the narrative

    def test_gram_unit_fails_safety(self, scorer, truth, resource):
        resource["dosage"][0]["doseAndRate"][0]["doseQuantity"]["unit"] = "g"
        s = scorer.score(resource, truth)
        assert not s.safety_passed
        assert any("unit" in f for f in s.failures)

    def test_overdose_fails_safety(self, scorer, truth, resource):
        resource["dosage"][0]["doseAndRate"][0]["doseQuantity"]["value"] = 8100
        s = scorer.score(resource, truth)
        assert not s.safety_passed
        assert any("exceeds ceiling" in f for f in s.failures)

    def test_missing_dosage_fails_constraints_and_safety(self, scorer, truth, resource):
        del resource["dosage"]
        s = scorer.score(resource, truth)
        assert not s.constraints_met and not s.safety_passed
        assert s.schema_valid and s.coding_correct

    def test_fabricated_dosage_fails_denied_case(self, scorer, cases):
        denied = next(c for c in cases if c.case_id == "ibuprofen-denied")
        truth = WriteAndValidateTask().ground_truth(denied)
        clean = build_expected(denied)
        assert scorer.score(clean, truth).passed

        fabricated = json.loads(
            hc_tools.build_medication_statement(
                denied.subject, "not-taken", "310965", "Ibuprofen 200 MG Oral Tablet",
                200, "mg", 3,
            )
        )["resource"]
        s = scorer.score(fabricated, truth)
        assert not s.safety_passed and not s.constraints_met
        assert any("fabricated" in f for f in s.failures)

    def test_stopped_medication_wrong_status_fails_constraints_only(self, scorer, cases):
        stopped = next(c for c in cases if c.case_id == "lisinopril-stopped-cough")
        truth = WriteAndValidateTask().ground_truth(stopped)
        resource = build_expected(stopped)

        resource["status"] = "active"

        s = scorer.score(resource, truth)
        assert not s.constraints_met
        assert s.schema_valid and s.coding_correct and s.safety_passed
        assert any("status" in f and "active" in f for f in s.failures)

    def test_stopped_medication_dosage_still_scores_safety(self, scorer, cases):
        stopped = next(c for c in cases if c.case_id == "lisinopril-stopped-cough")
        truth = WriteAndValidateTask().ground_truth(stopped)
        resource = build_expected(stopped)

        resource["dosage"][0]["timing"]["repeat"]["frequency"] = 2

        s = scorer.score(resource, truth)
        assert not s.safety_passed
        assert s.schema_valid and s.coding_correct and s.constraints_met
        assert any("administrations/day" in f for f in s.failures)

    def test_q6h_timing_is_understood(self, scorer, cases):
        # a hand-built resource may encode "every 6 hours" as period=6/h
        # rather than frequency=4/d; the scorer must treat them as equal
        tylenol = next(c for c in cases if c.case_id == "tylenol-brand-name")
        truth = WriteAndValidateTask().ground_truth(tylenol)
        resource = build_expected(tylenol)
        resource["dosage"][0]["timing"]["repeat"] = {
            "frequency": 1, "period": 6, "periodUnit": "h",
        }
        s = scorer.score(resource, truth)
        assert s.safety_passed, s.failures

    def test_summarize(self, scorer, truth, resource):
        good = scorer.score(resource, truth)
        bad = scorer.score({}, truth)
        summary = scorer.summarize([good, bad])
        assert summary["pass_rate"] == 0.5
        assert summary["schema_valid_rate"] == 0.5
        assert "PASS" in good.summary_line() and "FAIL" in bad.summary_line()


class CorrectStubAgent:
    """Answers with the tool-built correct resource for the current case."""

    name = "stub-correct"

    def __init__(self):
        self.case = None

    def run(self, prompt: str) -> str:
        return json.dumps(build_expected(self.case))


class GuessingStubAgent:
    """What a model-without-tools failure mode looks like: guessed code,
    R5 status from training data, no structured dosage."""

    name = "stub-guessing"

    def __init__(self):
        self.case = None

    def run(self, prompt: str) -> str:
        return json.dumps(
            {
                "resourceType": "MedicationStatement",
                "status": "recorded",
                "subject": {"reference": self.case.subject},
                "medicationCodeableConcept": {
                    "coding": [
                        {
                            "system": hc_tools.RXNORM_SYSTEM,
                            "code": "99999",
                            "display": "aspirin",
                        }
                    ]
                },
            }
        )


def test_offline_slice_correct_agent(cases):
    task, scorer, agent = WriteAndValidateTask(), WriteValidateScorer(), CorrectStubAgent()
    for case in cases:
        agent.case = case
        result = runner.run_case(task, agent, scorer, case)
        assert result.error is None
        assert result.score.passed, (case.case_id, result.score.failures)


def test_offline_slice_guessing_agent(cases):
    task, scorer, agent = WriteAndValidateTask(), WriteValidateScorer(), GuessingStubAgent()
    agent.case = cases[0]
    result = runner.run_case(task, agent, scorer, cases[0])
    assert result.error is None
    s = result.score
    assert not s.passed
    assert not s.schema_valid and not s.coding_correct and not s.safety_passed


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY")
def test_live_slice(tmp_path, cases):
    from groundeval.agents import HealthChainToolAgent

    init_tracing()
    # one dosed case and the denial case: exercises lookup, build, and
    # the don't-fabricate rule against the real agent loop
    subset = [c for c in cases if c.case_id in ("aspirin-active-daily", "ibuprofen-denied")]
    result = runner.run(
        task=WriteAndValidateTask(),
        agent=HealthChainToolAgent(),
        scorer=WriteValidateScorer(),
        cases=subset,
        out_dir=str(tmp_path),
    )
    assert result.summary["n_errors"] == 0
    record = json.loads(open(result.trace_path).read())
    assert record["summary"]["n_cases"] == 2

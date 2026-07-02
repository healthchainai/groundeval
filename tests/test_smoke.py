"""Smoke tests for the full eval slice.

`test_offline_slice` runs the whole pipeline with a stub agent (no network),
so the harness plumbing is always verifiable. `test_live_slice` runs the real
thing — fixture → Claude → score → trace — on the smallest fixture; it skips
when no ANTHROPIC_API_KEY is set.
"""

import json
import os

import pytest

from groundeval import runner
from groundeval.datasets import load_cases
from groundeval.scorers import DeterministicScorer
from groundeval.tasks import Medication, MedicationExtractionTask
from groundeval.tracing import init_tracing


class StubAgent:
    """Echoes the ground truth back, minus one medication, plus one fake."""

    name = "stub"

    def __init__(self, task):
        self.task = task
        self.case = None

    def run(self, prompt: str) -> str:
        truth = sorted(self.task.ground_truth(self.case), key=lambda m: m.rxnorm_code)
        meds = [{"name": m.name, "rxnorm_code": m.rxnorm_code} for m in truth[1:]]
        meds.append({"name": "madeupamab 10 MG", "rxnorm_code": "999999"})
        return json.dumps({"active_medications": meds})


def test_ground_truth_derivation():
    cases = load_cases()
    task = MedicationExtractionTask()
    truths = {c.case_id: task.ground_truth(c) for c in cases}
    # the fixture set spans 0..8 active medications by construction
    counts = sorted(len(t) for t in truths.values())
    assert counts == [0, 2, 3, 5, 7, 8]
    for truth in truths.values():
        for med in truth:
            assert med.rxnorm_code.isdigit()
            assert med.name


def test_scorer():
    scorer = DeterministicScorer()
    truth = {Medication("1", "aspirin 81 MG Oral Tablet"), Medication("2", "lisinopril 10 MG")}
    # one hit (name case-insensitive), one miss, one hallucination
    predicted = {Medication("1", "Aspirin 81 mg oral tablet"), Medication("3", "fake")}
    s = scorer.score(predicted, truth)
    assert not s.exact_match
    assert s.precision == 0.5 and s.recall == 0.5
    assert s.name_accuracy == 1.0
    assert s.false_positives == ["3"] and s.false_negatives == ["2"]

    empty = scorer.score(set(), set())
    assert empty.exact_match and empty.precision == 1.0 and empty.recall == 1.0


def test_offline_slice(tmp_path):
    cases = load_cases()[:2]
    task = MedicationExtractionTask()
    agent = StubAgent(task)
    results = []
    for case in cases:
        agent.case = case
        results.append(runner.run_case(task, agent, DeterministicScorer(), case))
    assert all(r.error is None for r in results)
    assert all(r.score.precision < 1.0 for r in results)  # stub always adds a fake med


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY")
def test_live_slice(tmp_path):
    from groundeval.agents import ClaudeAgent

    init_tracing()
    cases = load_cases()
    # smallest fixture: Clarissa466, 2 active medications
    case = next(c for c in cases if c.case_id.startswith("38f65650"))
    result = runner.run(
        task=MedicationExtractionTask(),
        agent=ClaudeAgent(),
        scorer=DeterministicScorer(),
        cases=[case],
        out_dir=str(tmp_path),
    )
    assert result.summary["n_errors"] == 0
    assert result.summary["mean_f1"] > 0
    record = json.loads(open(result.trace_path).read())
    assert record["summary"]["n_cases"] == 1

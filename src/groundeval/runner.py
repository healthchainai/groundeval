"""Run a task against an agent and score the results."""

import logging
import time
from dataclasses import dataclass, field

from langsmith import traceable

from groundeval.datasets import EvalCase
from groundeval.scorers import Score
from groundeval.tracing import write_local_trace

logger = logging.getLogger(__name__)


@dataclass
class CaseResult:
    case_id: str
    ground_truth: list
    predicted: list
    raw_output: str
    score: Score | None
    error: str | None
    duration_s: float


@dataclass
class RunResult:
    task: str
    agent: str
    scorer: str
    cases: list[CaseResult] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    trace_path: str = ""


@traceable(run_type="chain", name="eval-case")
def run_case(task, agent, scorer, case: EvalCase) -> CaseResult:
    """One case end-to-end: prompt → agent → parse → score.

    Agent and parse failures are recorded on the result instead of raised, so
    one bad case doesn't abort the run.
    """
    truth = task.ground_truth(case)
    start = time.monotonic()
    raw, predicted, score, error = "", set(), None, None
    try:
        raw = agent.run(task.prompt(case))
        predicted = task.parse_output(raw)
        score = scorer.score(predicted, truth)
    except Exception as e:  # noqa: BLE001 — any per-case failure becomes a recorded error
        error = f"{type(e).__name__}: {e}"
        logger.error("Case %s failed: %s", case.case_id, error)
    return CaseResult(
        case_id=case.case_id,
        ground_truth=sorted(truth, key=lambda m: m.rxnorm_code),
        predicted=sorted(predicted, key=lambda m: m.rxnorm_code),
        raw_output=raw,
        score=score,
        error=error,
        duration_s=round(time.monotonic() - start, 2),
    )


def summarize(results: list[CaseResult]) -> dict:
    scored = [r.score for r in results if r.score is not None]
    n = len(scored)
    mean = lambda xs: round(sum(xs) / n, 4) if n else 0.0  # noqa: E731
    return {
        "n_cases": len(results),
        "n_errors": len(results) - n,
        "exact_match_rate": mean([s.exact_match for s in scored]),
        "mean_precision": mean([s.precision for s in scored]),
        "mean_recall": mean([s.recall for s in scored]),
        "mean_f1": mean([s.f1 for s in scored]),
        "mean_name_accuracy": mean([s.name_accuracy for s in scored]),
    }


@traceable(run_type="chain", name="groundeval-run")
def run(task, agent, scorer, cases: list[EvalCase], out_dir: str = "runs") -> RunResult:
    result = RunResult(task=task.name, agent=agent.name, scorer=scorer.name)
    for case in cases:
        case_result = run_case(task, agent, scorer, case)
        result.cases.append(case_result)
        status = "ERROR" if case_result.error else f"f1={case_result.score.f1:.2f}"
        logger.info("Case %s: %s (%.1fs)", case.case_id, status, case_result.duration_s)
    result.summary = summarize(result.cases)

    record = {
        "task": result.task,
        "agent": result.agent,
        "scorer": result.scorer,
        "summary": result.summary,
        "cases": result.cases,
    }
    result.trace_path = str(write_local_trace(record, out_dir))
    return result

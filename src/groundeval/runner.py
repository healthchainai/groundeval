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
    raw, predicted, score, error = "", task.empty_output(), None, None
    try:
        raw = agent.run(task.prompt(case))
        predicted = task.parse_output(raw)
        score = scorer.score(predicted, truth)
    except Exception as e:  # noqa: BLE001 — any per-case failure becomes a recorded error
        error = f"{type(e).__name__}: {e}"
        logger.error("Case %s failed: %s", case.case_id, error)
    return CaseResult(
        case_id=case.case_id,
        ground_truth=task.record(truth),
        predicted=task.record(predicted),
        raw_output=raw,
        score=score,
        error=error,
        duration_s=round(time.monotonic() - start, 2),
    )


def summarize(results: list[CaseResult], scorer) -> dict:
    """Run-level counts plus whatever aggregation the scorer defines."""
    scored = [r.score for r in results if r.score is not None]
    return {
        "n_cases": len(results),
        "n_errors": len(results) - len(scored),
        **scorer.summarize(scored),
    }


@traceable(run_type="chain", name="groundeval-run")
def run(task, agent, scorer, cases: list[EvalCase], out_dir: str = "runs") -> RunResult:
    result = RunResult(task=task.name, agent=agent.name, scorer=scorer.name)
    for case in cases:
        case_result = run_case(task, agent, scorer, case)
        result.cases.append(case_result)
        status = "ERROR" if case_result.error else case_result.score.summary_line()
        logger.info("Case %s: %s (%.1fs)", case.case_id, status, case_result.duration_s)
    result.summary = summarize(result.cases, scorer)

    record = {
        "task": result.task,
        "agent": result.agent,
        "scorer": result.scorer,
        "summary": result.summary,
        "cases": result.cases,
    }
    result.trace_path = str(write_local_trace(record, out_dir))
    return result

"""Command-line entrypoint: `uv run groundeval`."""

import argparse
import logging
import sys

from groundeval import runner
from groundeval.agents import DEFAULT_MODEL, ClaudeAgent, HealthChainToolAgent
from groundeval.datasets import (
    DEFAULT_DATA_DIR,
    DEFAULT_WRITE_CASES_PATH,
    load_cases,
    load_write_cases,
)
from groundeval.scorers import DeterministicScorer, WriteValidateScorer
from groundeval.tasks import MedicationExtractionTask, WriteAndValidateTask
from groundeval.tracing import init_tracing


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="groundeval",
        description="Evaluate an agent on healthcare tasks over FHIR data.",
    )
    parser.add_argument(
        "--task",
        choices=["extraction", "write"],
        default="extraction",
        help="extraction: read meds from a bundle (v0.1); "
        "write: generate + validate a MedicationStatement with a tool-using agent",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="fixture bundle directory (extraction) or write-case JSON file (write)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Claude model id for the agent")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N cases")
    parser.add_argument("--out-dir", default="runs", help="directory for local run records")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    langsmith_on = init_tracing()
    if args.task == "write":
        cases = load_write_cases(args.data_dir or DEFAULT_WRITE_CASES_PATH)
        task, agent, scorer = (
            WriteAndValidateTask(),
            HealthChainToolAgent(model=args.model),
            WriteValidateScorer(),
        )
    else:
        cases = load_cases(args.data_dir or DEFAULT_DATA_DIR)
        task, agent, scorer = (
            MedicationExtractionTask(),
            ClaudeAgent(model=args.model),
            DeterministicScorer(),
        )
    if args.limit:
        cases = cases[: args.limit]

    result = runner.run(
        task=task,
        agent=agent,
        scorer=scorer,
        cases=cases,
        out_dir=args.out_dir,
    )

    print(f"\n{result.task} | agent={result.agent} | scorer={result.scorer}")
    print("-" * 72)
    for c in result.cases:
        if c.error:
            print(f"  {c.case_id}  ERROR: {c.error}")
        else:
            print(f"  {c.case_id}  {c.score.summary_line()}  ({c.duration_s}s)")
    print("-" * 72)
    for k, v in result.summary.items():
        print(f"  {k}: {v}")
    print(f"\nLocal run record: {result.trace_path}")
    print("LangSmith tracing:", "on" if langsmith_on else "off (set LANGSMITH_API_KEY to enable)")
    return 1 if result.summary["n_errors"] == result.summary["n_cases"] else 0


if __name__ == "__main__":
    sys.exit(main())

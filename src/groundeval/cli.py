"""Command-line entrypoint: `uv run groundeval`."""

import argparse
import logging
import sys

from groundeval import runner
from groundeval.agents import DEFAULT_MODEL, ClaudeAgent
from groundeval.datasets import DEFAULT_DATA_DIR, load_cases
from groundeval.scorers import DeterministicScorer
from groundeval.tasks import MedicationExtractionTask
from groundeval.tracing import init_tracing


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="groundeval",
        description="Evaluate an agent on healthcare tasks over FHIR data.",
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="fixture bundle directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Claude model id for the agent")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N cases")
    parser.add_argument("--out-dir", default="runs", help="directory for local run records")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    langsmith_on = init_tracing()
    cases = load_cases(args.data_dir)
    if args.limit:
        cases = cases[: args.limit]

    result = runner.run(
        task=MedicationExtractionTask(),
        agent=ClaudeAgent(model=args.model),
        scorer=DeterministicScorer(),
        cases=cases,
        out_dir=args.out_dir,
    )

    print(f"\n{result.task} | agent={result.agent} | scorer={result.scorer}")
    print("-" * 72)
    for c in result.cases:
        if c.error:
            print(f"  {c.case_id}  ERROR: {c.error}")
        else:
            s = c.score
            print(
                f"  {c.case_id}  exact={str(s.exact_match):5}  "
                f"P={s.precision:.2f} R={s.recall:.2f} F1={s.f1:.2f}  "
                f"names={s.name_accuracy:.2f}  ({c.duration_s}s)"
            )
    print("-" * 72)
    for k, v in result.summary.items():
        print(f"  {k}: {v}")
    print(f"\nLocal run record: {result.trace_path}")
    print("LangSmith tracing:", "on" if langsmith_on else "off (set LANGSMITH_API_KEY to enable)")
    return 1 if result.summary["n_errors"] == result.summary["n_cases"] else 0


if __name__ == "__main__":
    sys.exit(main())

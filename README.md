# groundeval

An open-source eval harness for healthcare AI agents, grounded in real clinical data standards.

Most LLM evals score models on text. Healthcare agents work over **structured clinical data** — FHIR resources with codes, statuses, and references — where "roughly right" isn't a passing grade. `groundeval` runs an agent against synthetic patient records, derives ground truth from the same data the agent sees, and scores the output deterministically.

FHIR is the testbed, not the point: the harness pattern (task defines prompt + ground truth + parsing; agent is swappable; scorer compares values) applies to any domain where answers are checkable against source data.

**v0.1** ships one vertical slice, end to end:

```
Synthea FHIR bundle  →  Claude (single call)  →  deterministic scorer  →  traced result
```

- **Task**: extract a patient's *active medications* from their FHIR bundle. Ground truth is derived from the bundle itself (every active `MedicationRequest`, identified by RxNorm code) — so the eval measures whether the model reads clinical data accurately, not whether it knows medicine.
- **Data**: six [Synthea](https://synthetichealth.github.io/synthea/) synthetic patients committed as fixtures (spanning 0–8 active medications, including a zero-medication patient to catch hallucinations). No PHI, no external data setup.
- **Scoring**: coded match on RxNorm gives precision/recall/F1; a second layer checks the extracted display names match exactly.
- **Tracing**: every run writes a local JSON record under `runs/`. Set `LANGSMITH_API_KEY` and runs also stream to LangSmith.

## Run it

Requires [uv](https://docs.astral.sh/uv/) and an Anthropic API key:

```bash
ANTHROPIC_API_KEY=sk-... uv run groundeval
```

That's the whole eval: loads the fixtures, runs `claude-opus-4-8` on each patient, prints per-case and aggregate scores, writes the run record.

Options: `--model` (any Claude model id), `--limit N` (first N cases), `--data-dir`, `--out-dir`.

Tests (`uv run pytest`) run the full pipeline offline against a stub agent; the live end-to-end test runs automatically when `ANTHROPIC_API_KEY` is set.

## How it's put together

```
src/groundeval/
  datasets.py   # fixture bundles → validated EvalCases (via healthchain.fhir)
  tasks.py      # a task = prompt + ground truth + output parsing
  agents.py     # an agent = name + run(prompt) -> text
  scorers.py    # a scorer = score(predicted, truth) -> Score
  runner.py     # orchestrates task × agent × scorer, aggregates
  tracing.py    # LangSmith when configured, local JSON always
  cli.py        # `uv run groundeval`
```

The seams are deliberate: a new task type is a new class in `tasks.py`, a new model is a new class in `agents.py`, and neither touches the runner. FHIR loading, validation, and resource extraction go through [HealthChain](https://github.com/healthchainai/HealthChain)'s FHIR helpers.

Fixtures are reproducible: `scripts/make_fixtures.py` trims full Synthea sample bundles (1–4 MB each) down to the resource types the task needs (~40–190 KB), keeping the data real but committable.

## Roadmap

1. More task types: medication reconciliation, summary generation, result interpretation
2. Model comparison across a second (open-weight) model
3. LLM-as-judge scoring for tasks without deterministic ground truth
4. Behavioural-consistency scoring: does the agent *reason* to answers it should *look up*?
5. Tool-using agents evaluated on the same tasks

## License

Apache 2.0

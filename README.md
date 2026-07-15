<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/healthchainai/groundeval/main/assets/brand/hero/readme-hero-dark.png">
    <img alt="groundeval" src="https://raw.githubusercontent.com/healthchainai/groundeval/main/assets/brand/hero/readme-hero-transparent.png" width="640">
  </picture>
</div>

An open-source eval harness for healthcare AI agents, grounded in real clinical data standards.

Most LLM evals score models on text. Healthcare agents work over **structured clinical data** — FHIR resources with codes, statuses, and references — where "roughly right" isn't a passing grade. `groundeval` runs an agent against synthetic patient records, derives ground truth from the same data the agent sees, and scores the output deterministically.

FHIR is the testbed, not the point: the harness pattern (task defines prompt + ground truth + parsing; agent is swappable; scorer compares values) applies to any domain where answers are checkable against source data.

Two task types so far, each a complete vertical slice:

**Extraction** (v0.1) — can a model *read* structured clinical data?

```
Synthea FHIR bundle  →  Claude (single call)  →  deterministic scorer  →  traced result
```

- **Task**: extract a patient's *active medications* from their FHIR bundle. Ground truth is derived from the bundle itself (every active `MedicationRequest`, identified by RxNorm code) — so the eval measures whether the model reads clinical data accurately, not whether it knows medicine.
- **Data**: six [Synthea](https://synthetichealth.github.io/synthea/) synthetic patients committed as fixtures (spanning 0–8 active medications, including a zero-medication patient to catch hallucinations). No PHI, no external data setup.
- **Scoring**: coded match on RxNorm gives precision/recall/F1; a second layer checks the extracted display names match exactly.

**Write-and-validate** (v0.2) — can a tool-using agent *write* it, correctly and safely?

```
clinical note  →  LangGraph agent ⇄ HealthChain FHIR tools  →  4-dimension scorer  →  traced result
```

- **Task**: given a clinical note excerpt ("takes two 200 mg ibuprofen tablets three times a day"), generate the FHIR `MedicationStatement` that documents it. Extraction a model can do from context; *generation* is where correctness has to be guaranteed, and where a bare model can't.
- **Agent**: a LangGraph agent whose tools wrap [HealthChain](https://github.com/healthchainai/HealthChain)'s FHIR operations (`groundeval/hc_tools.py`): look up RxNorm codes in a site catalog instead of guessing them, build the resource from flat arguments, validate before answering — validation errors come back as values the agent can read and fix.
- **Data**: seven committed cases with hand-authored ground truth — a brand-name lookup (Tylenol), per-administration dose arithmetic, a strength disambiguation (metoprolol 25 vs 100 MG ER), completed/stopped/denied statuses. The code catalog is real RxNorm content extracted from the Synthea fixtures.
- **Scoring**: deterministic, four independent pass/fail dimensions per case — **schema** (parses as R4B *and* satisfies the required status binding Pydantic doesn't enforce), **coding** (right RxNorm code and display), **constraints** (right patient, status reflects the narrative, dosage present exactly when documented), **safety** (encoded dose is the documented dose and the implied daily total is under the drug's ceiling; an unverifiable or fabricated dose fails). Schema-valid alone is table stakes — the eval bites on the other three.

Both slices: every run writes a local JSON record under `runs/`. Set `LANGSMITH_API_KEY` and runs (including the agent's tool calls) also stream to LangSmith.

## Run it

Requires [uv](https://docs.astral.sh/uv/) and an Anthropic API key:

```bash
ANTHROPIC_API_KEY=sk-... uv run groundeval                # extraction (v0.1)
ANTHROPIC_API_KEY=sk-... uv run groundeval --task write   # write-and-validate (v0.2)
```

Each loads its committed fixtures, runs `claude-opus-4-8` on every case, prints per-case and aggregate scores, and writes the run record.

Options: `--model` (any Claude model id), `--limit N` (first N cases), `--data-dir`, `--out-dir`.

Tests (`uv run pytest`) run the full pipeline offline against a stub agent; the live end-to-end test runs automatically when `ANTHROPIC_API_KEY` is set.

## How it's put together

```
src/groundeval/
  datasets.py   # committed fixtures → cases (FHIR bundles / write cases)
  tasks.py      # a task = prompt + ground truth + output parsing
  agents.py     # an agent = name + run(prompt) -> text (single-call or LangGraph)
  hc_tools.py   # HealthChain FHIR ops wrapped as agent tools — eval-free, liftable
  scorers.py    # a scorer = score(predicted, truth) -> Score + summarize(scores)
  runner.py     # orchestrates task × agent × scorer; value- and score-agnostic
  tracing.py    # LangSmith when configured, local JSON always
  cli.py        # `uv run groundeval`
```

The seams are deliberate: a new task type is a new class in `tasks.py` with its own scorer in `scorers.py`, a new agent is a new class in `agents.py`, and none of it touches the runner — v0.2 was added without changing it. FHIR loading, validation, and resource construction go through [HealthChain](https://github.com/healthchainai/HealthChain)'s FHIR helpers; `hc_tools.py` contains no eval logic so it can be promoted upstream into HealthChain's agent toolkit.

Fixtures are reproducible: `scripts/make_fixtures.py` trims full Synthea sample bundles (1–4 MB each) down to the resource types the task needs (~40–190 KB), and `scripts/make_terminology.py` derives the medication code catalog from those fixtures — the data stays real but committable. The friction hit while building against HealthChain is logged issue-ready in [FRICTION.md](FRICTION.md).

## Roadmap

1. More task types: medication reconciliation, summary generation, result interpretation
2. Model comparison across a second (open-weight) model
3. LLM-as-judge scoring for tasks without deterministic ground truth
4. Behavioural-consistency scoring: does the agent *reason* to answers it should *look up*?
5. ~~Tool-using agents evaluated on the same tasks~~ shipped in v0.2 for write-and-validate; next: run extraction through the tool-using agent for a with/without-tools comparison

## License

Apache 2.0

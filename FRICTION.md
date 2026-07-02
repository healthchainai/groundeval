# HealthChain friction log — groundeval

Raw notes from building groundeval against `healthchain.fhir` (PyPI 0.15.0).
Two channels, per the build spec: channel 1 (data/scoring friction) from v0.1,
channel 2 (agent-tool friction) from the v0.2 write-and-validate slice.
Channel 1 entries have been filed to `healthchainai/HealthChain`: stub 1 →
[#232](https://github.com/healthchainai/HealthChain/issues/232), stub 2 →
[#233](https://github.com/healthchainai/HealthChain/issues/233), stub 3 →
[#234](https://github.com/healthchainai/HealthChain/issues/234).
Channel 2 stubs (4–8 below) are not yet filed.

## Channel 1 — data / scoring friction

### Issue stub 1: No public way to load a FHIR bundle from a file (or dict) that surfaces validation errors

`healthchain.fhir` has no `load_bundle(path)`. The closest public helper,
`create_resource_from_dict(dict, "Bundle")`, catches every exception, logs it,
and returns `None` — so a harness (or any pipeline that must trust its inputs)
can't tell *why* a bundle failed validation without scraping logs, and a `None`
return is easy to ignore silently. groundeval wraps it and re-raises a generic
error, losing the underlying Pydantic detail. There *is* a Synthea file loader,
but it lives in `healthchain.sandbox.loaders` coupled to the SandboxClient
registry, not usable as a plain function.

Suggested API:

```python
from healthchain.fhir import load_bundle
bundle = load_bundle("patient.json")           # Path | str | dict
# raises FHIRValidationError (wrapping the pydantic error) on invalid input
```

Or minimally: a `raise_on_error: bool = False` parameter on
`create_resource_from_dict`.

### Issue stub 2: No public helper to read a medication's identity (code / display / system) off a resource

Deriving "active medications with RxNorm codes" from a bundle required manual
traversal: `get_resources(bundle, "MedicationRequest")`, then
`req.medicationCodeableConcept.coding[0].code / .display`, plus hand-rolled
status filtering, None-guards, and dedup by code. The only flattening code that
exists (`dataframe._flatten_medications`) is private, dict-based, and shaped
for ML indicator columns (`medication_{code}_{display}: 1`), not for reading
values back out. The `healthchain.fhir` API is write-rich (many `create_*`
helpers) but read-poor — extraction beyond "give me the resources" is on the
caller.

Suggested API:

```python
from healthchain.fhir import get_medications
meds = get_medications(bundle, status="active")
# -> [MedicationEntry(code="243670", display="aspirin 81 MG Oral Tablet",
#                     system="http://www.nlm.nih.gov/research/umls/rxnorm",
#                     status="active", source_type="MedicationRequest"), ...]
```

covering both `medicationCodeableConcept` and (resolving) `medicationReference`,
across MedicationRequest / MedicationStatement. A generic
`get_codings(resource, field)` would serve the same need for Condition,
AllergyIntolerance, etc.

### Issue stub 3: `medicationReference` requests are second-class — no reference resolution within a bundle

Synthea bundles mix inline `medicationCodeableConcept` with
`medicationReference` pointing at contained `Medication` resources. Nothing in
`healthchain.fhir` resolves a reference to its target resource within a bundle,
so any medication logic that wants to be correct on real-world data needs
hand-written urn/uuid lookup. v0.1 sidestepped this (no *active* request in the
fixture set uses a reference — verified, not assumed), but the first dataset
where that isn't true breaks ground-truth derivation.

Suggested API:

```python
from healthchain.fhir import resolve_reference
med = resolve_reference(bundle, request.medicationReference)  # -> Medication | None
```

### Non-issue observations (no stub needed)

- `get_resources(bundle, "MedicationRequest")` is exactly the right seam and
  worked first try; string-or-type dispatch is pleasant.
- Round-tripping raw Synthea JSON through validation was clean — no timezone or
  field-name fights (the R4B models handled everything in the sample set).

## Channel 2 — agent tools friction

From building `groundeval/hc_tools.py` for v0.2: the first code to expose
`healthchain.fhir` operations as tools for a tool-calling agent (LangGraph +
Claude). Everything below was hit while hand-rolling three tools — code
lookup, resource construction, validation — for a write-and-validate eval.
Together these stubs are a spec for HealthChain's agent toolkit (README
roadmap item 5).

### Issue stub 4: No agent-callable validation — errors are swallowed, not returned

An agent loop lives or dies on validation feedback: the model builds a
resource, the tool must answer "valid, or here is exactly what's wrong" *as a
value* the model can read and correct. `healthchain.fhir` has no
`validate_resource()` at all — validation only happens implicitly inside
`create_resource_from_dict`, which catches every exception, logs, and returns
`None`. A `None` gives an agent nothing to self-correct on. groundeval had to
hand-roll a two-layer validator (Pydantic model validation + required-binding
checks, issues returned as a JSON list). This is the single most important
tool in the write loop and HealthChain offers no entry point for it.

Suggested API:

```python
from healthchain.fhir import validate_resource
report = validate_resource(resource_or_dict, resource_type="MedicationStatement")
# -> ValidationReport(valid: bool, issues: list[str])  — JSON-serializable,
#    never raises; includes required ValueSet binding checks (see stub 5)
```

### Issue stub 5: `create_medication_statement` defaults to `status="recorded"` — not a valid R4B status

The R4B `MedicationStatement.status` required binding is `active | completed |
entered-in-error | intended | stopped | on-hold | unknown | not-taken`;
"recorded" is R5 vocabulary. The helper's default therefore produces a
spec-invalid resource out of the box — and it flows through silently because
`fhir.resources` keeps enum values in `json_schema_extra` without enforcing
them (verified: `status="recorded"` passes `model_validate`). Any downstream
system that checks bindings (a FHIR server, an IG validator) will reject what
HealthChain builds by default. The tool adapter had to own the value-set check
itself.

Suggested fix: version-aware status defaults + an explicit binding check in
the `create_*` helpers (raise or return an error on a status outside the
target version's value set), ideally shared with `validate_resource` (stub 4).

### Issue stub 6: `create_medication_statement` cannot express dosage

A MedicationStatement without dose and frequency is clinically inert — nothing
downstream can reconcile or safety-check it. The helper only takes
`subject/status/code/display/system`, so encoding "81 mg once daily" meant
hand-building the four-level R4B tree (`Dosage → Timing → TimingRepeat` and
`Dosage → DosageDoseAndRate → Quantity`) and assigning `ms.dosage = [...]`.
That is exactly the nesting a flat-argument agent tool exists to hide, and
every write-path user (agent or not) will need it.

Suggested API (mirroring the helper's flat-kwarg style):

```python
create_medication_statement(..., dose_value=81, dose_unit="mg",
                            frequency_per_day=1)
# or an element helper: create_dosage(dose_value, dose_unit,
#                                     frequency_per_day, text=None) -> Dosage
```

### Issue stub 7: No terminology lookup surface — coding helpers only write codes the caller already has

`create_single_codeable_concept` / `add_coding_to_codeable_concept` assume the
correct code is in hand. Nothing in HealthChain maps a medication name (brand,
ingredient, strength) to candidate codings — not even against a local
formulary table. For an agent this is the whole game: the defensible reason to
give a model a tool instead of trusting parametric memory is that codes get
*looked up*, not recalled. groundeval ships its own token-match lookup over a
committed RxNorm table derived from Synthea data; HealthChain gave it nothing
to wrap.

Suggested API: a small lookup protocol with a local-table implementation, so
sites can plug a real terminology service later:

```python
from healthchain.terminology import LocalCodeLookup
lookup = LocalCodeLookup.from_json("formulary.json")   # or a TerminologyService impl
lookup.search("metoprolol succinate 25", system=RXNORM)  # -> [Coding], ranked
```

### Issue stub 8: No first-class "expose as agent tool" path

Turning three `healthchain.fhir` operations into agent tools required
hand-writing, per tool: a flat scalar argument schema (models emit flat JSON
reliably; full FHIR resources as arguments are too nested to generate
dependably), JSON-string returns, errors-as-return-values instead of
exceptions, and agent-facing docstrings that become the tool descriptions.
None of that is groundeval-specific — it is the adapter layer any
agent-framework user rebuilds. HealthChain's create-helpers are already
tool-shaped (flat kwargs); they are just not packaged as tools.

Suggested API:

```python
from healthchain.tools import fhir_tools
tools = fhir_tools(resources=["MedicationStatement"], lookup=my_lookup)
# framework-agnostic callables with schemas; .as_langchain() / .as_mcp() views
```

### Non-issue observations (no stub needed)

- The flat-kwarg style of `create_medication_statement` mapped 1:1 onto a
  tool argument schema — the write-rich API is *almost* an agent toolkit
  already; stubs 4–8 are the gap between "almost" and "is".
- Assigning hand-built elements onto a created resource (`ms.dosage = [...]`)
  round-tripped through validation cleanly; no fights with the R4B models.

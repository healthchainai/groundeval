# HealthChain friction log — groundeval v0.1

Raw notes from building v0.1 against `healthchain.fhir` (PyPI 0.15.0). Two
channels, per the build spec. Channel 1 entries are written issue-ready for
`healthchainai/HealthChain`; none have been filed yet.

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

N/A in v0.1 — no tool-using agent. The v0.1 agent under test is a toolless
single Claude call, so "places an agent can't easily call HealthChain in" was
not exercised. An empty section here means *not tested*, not *no problems*.

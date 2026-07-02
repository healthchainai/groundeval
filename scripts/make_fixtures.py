"""Build the committed Synthea fixture set from raw Synthea FHIR output.

Full Synthea patient bundles are 1-4 MB each, which is too large to commit or
to hand to a model. This script trims each selected patient's bundle down to
the resource types the extraction task needs (plus clinical context), converts
it to a `collection` bundle, and validates the result before writing it to
data/synthea/.

Usage:
    uv run python scripts/make_fixtures.py <path-to-synthea-fhir-dir>

Source data: https://synthetichealth.github.io/synthea-sample-data/downloads/latest/synthea_sample_data_fhir_latest.zip
"""

import json
import sys
from pathlib import Path

from healthchain.fhir import create_resource_from_dict

# Patient files selected for a spread of active-medication counts (0 to 8),
# including a zero-active patient so the scorer is exercised against
# hallucinated medications. Moderate bundle sizes only (<= 60 MedicationRequests).
SELECTED = [
    "Alena861_Alina705_Bayer639_a41e603d-0d7c-c89f-3be7-74117adc6390.json",  # 0 active
    "Clarissa466_Feeney44_38f65650-9735-8541-322d-387417329eb2.json",  # 2 active
    "Felicitas300_Donette997_Hayes766_fcf99efa-d698-1818-b988-57b1680c789e.json",  # 3 active
    "Gerda633_Williamson769_11fb6b9d-609d-25c1-1859-a5af1c48da45.json",  # 5 active
    "Chiquita638_Vandervort697_269c9001-961e-5c04-c250-442ec7c276c5.json",  # 7 active
    "Eilene124_Farrell962_be2cd7b7-6d1b-b446-deb4-7b850d845064.json",  # 8 active
]

# Resource types kept in fixtures: the task target (MedicationRequest, plus
# Medication so medicationReference entries resolve) and enough clinical
# context to make the bundle realistically noisy.
KEEP_TYPES = {"Patient", "MedicationRequest", "Medication", "Condition", "AllergyIntolerance"}

OUT_DIR = Path(__file__).parent.parent / "data" / "synthea"


def trim_bundle(bundle: dict) -> dict:
    entries = [
        {"fullUrl": e.get("fullUrl"), "resource": e["resource"]}
        for e in bundle.get("entry", [])
        if e["resource"]["resourceType"] in KEEP_TYPES
    ]
    return {"resourceType": "Bundle", "type": "collection", "entry": entries}


def main(synthea_dir: str) -> None:
    src = Path(synthea_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in SELECTED:
        raw = json.loads((src / name).read_text())
        trimmed = trim_bundle(raw)
        if create_resource_from_dict(trimmed, "Bundle") is None:
            raise ValueError(f"Trimmed bundle failed FHIR validation: {name}")
        out = OUT_DIR / name
        out.write_text(json.dumps(trimmed, indent=2))
        print(f"{name}: {len(trimmed['entry'])} entries, {out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])

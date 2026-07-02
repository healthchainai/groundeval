"""Build the committed medication terminology table from the Synthea fixtures.

The write-and-validate task needs an offline stand-in for a terminology
service: the agent looks up RxNorm codes instead of guessing them from
parametric memory. This script collects every distinct medication coding that
appears in the committed fixture bundles (any status — stopped and completed
medications carry codes just as real as active ones) and writes them to
data/terminology/rxnorm_medications.json. Codes and display names are real
RxNorm content from Synthea, not hand-typed.

Usage:
    uv run python scripts/make_terminology.py
"""

import json
from pathlib import Path

from healthchain.fhir import create_resource_from_dict, get_resources

DATA_DIR = Path(__file__).parent.parent / "data"
FIXTURE_DIR = DATA_DIR / "synthea"
OUT_PATH = DATA_DIR / "terminology" / "rxnorm_medications.json"


def main() -> None:
    codings: dict[str, dict] = {}
    for f in sorted(FIXTURE_DIR.glob("*.json")):
        bundle = create_resource_from_dict(json.loads(f.read_text()), "Bundle")
        if bundle is None:
            raise ValueError(f"{f.name} failed FHIR validation")
        for req in get_resources(bundle, "MedicationRequest"):
            concept = req.medicationCodeableConcept
            if concept is None or not concept.coding:
                continue
            c = concept.coding[0]
            codings[str(c.code)] = {
                "code": str(c.code),
                "display": str(c.display),
                "system": str(c.system),
            }

    table = sorted(codings.values(), key=lambda x: x["display"].lower())
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(table, indent=2) + "\n")
    print(f"{OUT_PATH}: {len(table)} distinct medication codings")


if __name__ == "__main__":
    main()

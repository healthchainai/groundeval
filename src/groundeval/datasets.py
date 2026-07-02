"""Load committed FHIR fixture bundles into eval cases."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from fhir.resources.R4B.bundle import Bundle
from healthchain.fhir import create_resource_from_dict

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "synthea"


@dataclass
class EvalCase:
    """One patient bundle to evaluate against.

    `bundle` is the validated FHIR object used for ground-truth derivation;
    `bundle_json` is the raw fixture text handed to the agent.
    """

    case_id: str
    bundle: Bundle
    bundle_json: str


def load_cases(data_dir: Path | str = DEFAULT_DATA_DIR) -> list[EvalCase]:
    """Load and validate every fixture bundle in `data_dir`."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No fixture bundles found in {data_dir}")

    cases = []
    for f in files:
        raw = f.read_text()
        bundle = create_resource_from_dict(json.loads(raw), "Bundle")
        if bundle is None:
            # create_resource_from_dict logs the validation error and returns
            # None; surface a hard failure so a bad fixture can't silently
            # drop out of the eval set.
            raise ValueError(f"{f.name} failed FHIR validation (see log above)")
        # case_id = the patient UUID portion of the Synthea filename
        case_id = f.stem.split("_")[-1]
        cases.append(EvalCase(case_id=case_id, bundle=bundle, bundle_json=raw))
    logger.info("Loaded %d cases from %s", len(cases), data_dir)
    return cases

"""Load committed FHIR fixture bundles into eval cases."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from fhir.resources.R4B.bundle import Bundle
from healthchain.fhir import create_resource_from_dict

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "synthea"
DEFAULT_WRITE_CASES_PATH = (
    Path(__file__).parent.parent.parent / "data" / "write_cases" / "medication_statements.json"
)


@dataclass
class EvalCase:
    """One patient bundle to evaluate against.

    `bundle` is the validated FHIR object used for ground-truth derivation;
    `bundle_json` is the raw fixture text handed to the agent.
    """

    case_id: str
    bundle: Bundle
    bundle_json: str


@dataclass
class WriteCase:
    """One write-and-validate case: a clinical note excerpt to encode as FHIR.

    `expected` is the scoring ground truth (coding, status, dosage, safety
    ceiling) straight from the fixture file; the task turns it into a typed
    expectation and it is never shown to the agent.
    """

    case_id: str
    subject: str
    input_text: str
    expected: dict


def load_write_cases(path: Path | str = DEFAULT_WRITE_CASES_PATH) -> list[WriteCase]:
    """Load and sanity-check the committed write-and-validate fixture file."""
    path = Path(path)
    payload = json.loads(path.read_text())
    cases = []
    for raw in payload["cases"]:
        expected = raw["expected"]
        missing = {"rxnorm_code", "display", "status", "dosage", "max_daily_dose"} - set(expected)
        if missing:
            raise ValueError(f"Case {raw['case_id']}: expected block missing {sorted(missing)}")
        cases.append(
            WriteCase(
                case_id=raw["case_id"],
                subject=raw["subject"],
                input_text=raw["input"],
                expected=expected,
            )
        )
    logger.info("Loaded %d write cases from %s", len(cases), path)
    return cases


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

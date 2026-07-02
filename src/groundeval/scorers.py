"""Deterministic scoring of extracted values against ground truth."""

from dataclasses import dataclass, field

from groundeval.tasks import Medication


@dataclass
class Score:
    """Per-case score for a set-extraction task.

    Coded match (RxNorm code) drives precision/recall/F1 — the code is the
    unambiguous identity of a medication. `name_accuracy` then checks that,
    for each correctly coded medication, the extracted display name also
    matches ground truth exactly (case/whitespace-insensitive).
    """

    exact_match: bool
    precision: float
    recall: float
    f1: float
    name_accuracy: float
    true_positives: list[str] = field(default_factory=list)
    false_positives: list[str] = field(default_factory=list)
    false_negatives: list[str] = field(default_factory=list)


def _normalize(name: str) -> str:
    return " ".join(name.lower().split())


class DeterministicScorer:
    name = "deterministic"

    def score(self, predicted: set[Medication], truth: set[Medication]) -> Score:
        pred_codes = {m.rxnorm_code for m in predicted}
        truth_codes = {m.rxnorm_code for m in truth}

        tp = pred_codes & truth_codes
        fp = pred_codes - truth_codes
        fn = truth_codes - pred_codes

        precision = len(tp) / len(pred_codes) if pred_codes else (1.0 if not truth_codes else 0.0)
        recall = len(tp) / len(truth_codes) if truth_codes else 1.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        truth_names = {m.rxnorm_code: _normalize(m.name) for m in truth}
        pred_names = {m.rxnorm_code: _normalize(m.name) for m in predicted}
        name_matches = [pred_names[code] == truth_names[code] for code in tp]
        name_accuracy = sum(name_matches) / len(name_matches) if name_matches else 1.0

        return Score(
            exact_match=(pred_codes == truth_codes),
            precision=precision,
            recall=recall,
            f1=f1,
            name_accuracy=name_accuracy,
            true_positives=sorted(tp),
            false_positives=sorted(fp),
            false_negatives=sorted(fn),
        )

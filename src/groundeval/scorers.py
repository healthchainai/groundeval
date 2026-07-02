"""Deterministic scoring of extracted values against ground truth.

A scorer owns its Score shape end to end: `score()` produces one per case,
`summarize()` aggregates them, and `Score.summary_line()` renders one for
logs. The runner never touches a concrete score's fields, so a scorer with a
different shape (graded, rubric-based, ...) plugs in without core changes.
"""

from dataclasses import dataclass, field

from groundeval.tasks import Medication


class Score:
    """Base type for per-case scores. Subclasses define their own fields."""

    def summary_line(self) -> str:
        """One-line human-readable rendering for logs and CLI output."""
        raise NotImplementedError


@dataclass
class MedicationMatchScore(Score):
    """Per-case score for medication set extraction.

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

    def summary_line(self) -> str:
        return (
            f"exact={str(self.exact_match):5}  "
            f"P={self.precision:.2f} R={self.recall:.2f} F1={self.f1:.2f}  "
            f"names={self.name_accuracy:.2f}"
        )


def _normalize(name: str) -> str:
    return " ".join(name.lower().split())


class DeterministicScorer:
    name = "deterministic"

    def score(self, predicted: set[Medication], truth: set[Medication]) -> MedicationMatchScore:
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

        return MedicationMatchScore(
            exact_match=(pred_codes == truth_codes),
            precision=precision,
            recall=recall,
            f1=f1,
            name_accuracy=name_accuracy,
            true_positives=sorted(tp),
            false_positives=sorted(fp),
            false_negatives=sorted(fn),
        )

    def summarize(self, scores: list[MedicationMatchScore]) -> dict:
        n = len(scores)
        mean = lambda xs: round(sum(xs) / n, 4) if n else 0.0  # noqa: E731
        return {
            "exact_match_rate": mean([s.exact_match for s in scores]),
            "mean_precision": mean([s.precision for s in scores]),
            "mean_recall": mean([s.recall for s in scores]),
            "mean_f1": mean([s.f1 for s in scores]),
            "mean_name_accuracy": mean([s.name_accuracy for s in scores]),
        }

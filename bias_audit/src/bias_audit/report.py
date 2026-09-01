from dataclasses import dataclass
from typing import Dict, Sequence, Union

import numpy as np

from .metrics import SplitResult, effectiveness_score, evaluate_predictions
from .types import Classifier, LabeledExample


@dataclass
class AuditReport:
    label_names: Sequence[str]
    overall: SplitResult
    per_group: Dict[str, SplitResult]
    max_group_f1_gap: float
    effectiveness_score: float

    def summary(self) -> str:
        lines = [
            f"n={self.overall.n_examples}  "
            f"F1_macro={self.overall.f1_macro:.3f}  F1_micro={self.overall.f1_micro:.3f}",
            "",
            "Per-label F1:",
        ]
        for label, f1 in self.overall.per_label_f1.items():
            lines.append(f"  {label}: {f1:.3f}")
        if self.per_group:
            lines.append("")
            lines.append("Per-group F1_macro:")
            for group, result in sorted(self.per_group.items()):
                lines.append(f"  {group} (n={result.n_examples}): {result.f1_macro:.3f}")
            lines.append("")
            lines.append(f"Max group F1_macro gap: {self.max_group_f1_gap:.3f}")
            lines.append(
                f"Effectiveness score (F1_macro - max group gap): {self.effectiveness_score:.3f}"
            )
        return "\n".join(lines)


def _label_matrix(examples: Sequence[LabeledExample]) -> np.ndarray:
    return np.array([list(ex.labels) for ex in examples])


def audit(
    classifier: Classifier,
    examples: Sequence[LabeledExample],
    label_names: Sequence[str],
    thresholds: Union[float, Sequence[float]] = 0.5,
) -> AuditReport:
    """Score `classifier` against `examples` overall and broken down by `.group`.

    Returns an AuditReport with per-label F1, a per-group F1 breakdown, the max
    F1 gap between any two groups, and an effectiveness score (overall F1_macro
    minus that gap) that rewards accuracy but penalizes an uneven spread across
    groups - the thing you actually want to know when a group is a proxy for
    something the classifier shouldn't be keying off of (e.g. source style
    standing in for outlet lean).
    """
    if not examples:
        raise ValueError("audit() requires at least one labeled example")

    texts = [ex.text for ex in examples]
    y_true = _label_matrix(examples)
    probs = np.asarray(classifier.predict_proba(texts))
    thresh_arr = (
        np.full(len(label_names), thresholds, dtype=float)
        if np.isscalar(thresholds)
        else np.asarray(thresholds, dtype=float)
    )
    y_pred = (probs >= thresh_arr).astype(int)

    overall = evaluate_predictions(y_true, y_pred, label_names)

    by_group: Dict[str, list] = {}
    for i, ex in enumerate(examples):
        by_group.setdefault(ex.group, []).append(i)

    per_group: Dict[str, SplitResult] = {}
    for group, idx in sorted(by_group.items()):
        per_group[group] = evaluate_predictions(y_true[idx], y_pred[idx], label_names)

    gap, eff = (
        effectiveness_score(overall.f1_macro, {g: r.f1_macro for g, r in per_group.items()})
        if len(per_group) > 1
        else (0.0, overall.f1_macro)
    )

    return AuditReport(
        label_names=label_names,
        overall=overall,
        per_group=per_group,
        max_group_f1_gap=gap,
        effectiveness_score=eff,
    )

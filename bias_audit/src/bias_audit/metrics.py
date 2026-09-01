from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
from sklearn.metrics import f1_score


@dataclass
class SplitResult:
    n_examples: int
    f1_micro: float
    f1_macro: float
    per_label_f1: Dict[str, float]
    confusion: Dict[str, Dict[str, int]]


def label_confusion(
    y_true: np.ndarray, y_pred: np.ndarray, label_names: Sequence[str]
) -> Dict[str, Dict[str, int]]:
    """Counts, for each true label, which other labels got predicted alongside it.

    Only cross-label pairs are counted (a correct match on the same label is not a
    confusion) - useful for spotting which techniques the model conflates.
    """
    confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for true_row, pred_row in zip(y_true, y_pred):
        true_labels = [label_names[i] for i, v in enumerate(true_row) if v == 1]
        pred_labels = [label_names[i] for i, v in enumerate(pred_row) if v == 1]
        for t in true_labels:
            for p in pred_labels:
                if t != p:
                    confusion[t][p] += 1
    return {t: dict(p) for t, p in confusion.items()}


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, label_names: Sequence[str]
) -> SplitResult:
    return SplitResult(
        n_examples=len(y_true),
        f1_micro=float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        per_label_f1={
            label: float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0))
            for i, label in enumerate(label_names)
        },
        confusion=label_confusion(y_true, y_pred, label_names),
    )


def effectiveness_score(
    overall_f1_macro: float, per_group_f1_macro: Dict[str, float]
) -> Tuple[float, float]:
    """Returns (max_group_gap, effectiveness_score).

    effectiveness_score = overall_f1_macro - max_group_gap: a single number that
    rewards accuracy but penalizes an uneven spread across groups, so a classifier
    can't game it by being accurate on average while badly biased on one group.
    """
    if not per_group_f1_macro:
        return 0.0, overall_f1_macro
    values = list(per_group_f1_macro.values())
    gap = max(values) - min(values)
    return gap, overall_f1_macro - gap

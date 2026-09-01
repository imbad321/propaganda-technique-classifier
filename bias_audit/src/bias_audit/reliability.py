from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import cohen_kappa_score


@dataclass
class LabelReliability:
    per_label_kappa: Dict[str, float]
    mean_kappa: float
    exact_match_rate: float
    mean_jaccard: float

    def summary(self) -> str:
        lines = [f"{'label':<28}{'kappa':>8}"]
        for label, kappa in self.per_label_kappa.items():
            lines.append(f"{label:<28}{kappa:>8.3f}")
        lines.append("")
        lines.append(f"Mean per-label kappa: {self.mean_kappa:.3f}")
        lines.append(f"Exact full-vector agreement: {self.exact_match_rate * 100:.1f}%")
        lines.append(f"Mean per-sentence Jaccard overlap: {self.mean_jaccard:.3f}")
        return "\n".join(lines)


def label_reliability(
    labels_a: np.ndarray, labels_b: np.ndarray, label_names: Sequence[str]
) -> LabelReliability:
    """Cohen's kappa (per label) between two independent labelings of the same examples.

    Use this to sanity-check a hand-labeled evaluation set against a second annotator
    before trusting an audit() result built on it - a low kappa on one label means
    that label's ground truth (and any bias-audit finding that leans on it) is
    noisier than the others, independent of how good the classifier is.
    """
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    if labels_a.shape != labels_b.shape:
        raise ValueError(f"shape mismatch: {labels_a.shape} vs {labels_b.shape}")

    per_label_kappa = {}
    for i, label in enumerate(label_names):
        a, b = labels_a[:, i], labels_b[:, i]
        if a.std() == 0 and b.std() == 0:
            kappa = 1.0 if (a == b).all() else 0.0
        else:
            kappa = float(cohen_kappa_score(a, b))
        per_label_kappa[label] = kappa

    exact_match = float((labels_a == labels_b).all(axis=1).mean())

    any_overlap = ((labels_a + labels_b) > 0).sum(axis=1)
    both = ((labels_a == 1) & (labels_b == 1)).sum(axis=1)
    jaccard = np.divide(
        both, any_overlap, out=np.ones_like(both, dtype=float), where=any_overlap != 0
    )

    return LabelReliability(
        per_label_kappa=per_label_kappa,
        mean_kappa=float(np.mean(list(per_label_kappa.values()))),
        exact_match_rate=exact_match,
        mean_jaccard=float(jaccard.mean()),
    )

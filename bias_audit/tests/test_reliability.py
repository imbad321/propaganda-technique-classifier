import numpy as np
import pytest

from bias_audit.reliability import label_reliability

LABELS = ["a", "b"]


def test_perfect_agreement_gives_kappa_one():
    labels = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    result = label_reliability(labels, labels, LABELS)
    assert result.per_label_kappa == {"a": 1.0, "b": 1.0}
    assert result.mean_kappa == 1.0
    assert result.exact_match_rate == 1.0
    assert result.mean_jaccard == 1.0


def test_constant_column_both_labelers_agree_scores_one():
    # every row is 0 for label "a" for both labelers - no variance, but they agree
    a = np.array([[0, 1], [0, 0], [0, 1]])
    b = np.array([[0, 1], [0, 1], [0, 1]])
    result = label_reliability(a, b, LABELS)
    assert result.per_label_kappa["a"] == 1.0


def test_constant_column_labelers_disagree_scores_zero():
    a = np.array([[0, 1], [0, 0]])
    b = np.array([[1, 1], [1, 0]])
    result = label_reliability(a, b, LABELS)
    assert result.per_label_kappa["a"] == 0.0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        label_reliability(np.array([[1, 0]]), np.array([[1, 0], [0, 1]]), LABELS)

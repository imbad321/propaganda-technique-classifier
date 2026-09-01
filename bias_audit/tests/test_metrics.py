import numpy as np
import pytest

from bias_audit.metrics import effectiveness_score, evaluate_predictions, label_confusion

LABELS = ["a", "b"]


def test_evaluate_predictions_perfect_match():
    y = np.array([[1, 0], [0, 1], [1, 1]])
    result = evaluate_predictions(y, y, LABELS)
    assert result.n_examples == 3
    assert result.f1_macro == 1.0
    assert result.f1_micro == 1.0
    assert result.per_label_f1 == {"a": 1.0, "b": 1.0}


def test_evaluate_predictions_no_overlap_scores_zero():
    y_true = np.array([[1, 0], [0, 1]])
    y_pred = np.array([[0, 1], [1, 0]])
    result = evaluate_predictions(y_true, y_pred, LABELS)
    assert result.f1_macro == 0.0


def test_label_confusion_counts_cross_label_pairs_only():
    # row 0: true=a, pred=a+b -> one cross pair (a, b)
    # row 1: true=pred=b -> no cross pair
    y_true = np.array([[1, 0], [0, 1]])
    y_pred = np.array([[1, 1], [0, 1]])
    assert label_confusion(y_true, y_pred, LABELS) == {"a": {"b": 1}}


def test_effectiveness_score_penalizes_group_spread():
    gap, score = effectiveness_score(0.6, {"x": 0.7, "y": 0.5})
    assert gap == pytest.approx(0.2)
    assert score == pytest.approx(0.4)


def test_effectiveness_score_no_groups_returns_overall():
    gap, score = effectiveness_score(0.6, {})
    assert gap == 0.0
    assert score == 0.6

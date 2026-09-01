import numpy as np
import pytest

from bias_audit.report import audit
from bias_audit.types import LabeledExample

LABELS = ["a", "b"]


class DictClassifier:
    """Fake Classifier that returns a fixed probability row per input text."""

    def __init__(self, probs_by_text):
        self.probs_by_text = probs_by_text

    def predict_proba(self, texts):
        return np.array([self.probs_by_text[t] for t in texts])


def test_audit_reports_per_group_gap_and_effectiveness_score():
    examples = [
        LabeledExample(text="t1", labels=[1, 0], group="x"),
        LabeledExample(text="t2", labels=[0, 1], group="x"),
        LabeledExample(text="t3", labels=[1, 0], group="y"),
        LabeledExample(text="t4", labels=[0, 1], group="y"),
    ]
    # group "x" predicted perfectly, group "y" predicted exactly backwards
    classifier = DictClassifier(
        {"t1": [1, 0], "t2": [0, 1], "t3": [0, 1], "t4": [1, 0]}
    )

    report = audit(classifier, examples, LABELS)

    assert report.overall.f1_macro == pytest.approx(0.5)
    assert report.per_group["x"].f1_macro == pytest.approx(1.0)
    assert report.per_group["y"].f1_macro == pytest.approx(0.0)
    assert report.max_group_f1_gap == pytest.approx(1.0)
    assert report.effectiveness_score == pytest.approx(-0.5)
    assert "Effectiveness score" in report.summary()


def test_audit_single_group_has_no_gap_penalty():
    examples = [
        LabeledExample(text="t1", labels=[1, 0], group="only"),
        LabeledExample(text="t2", labels=[0, 1], group="only"),
    ]
    classifier = DictClassifier({"t1": [1, 0], "t2": [0, 1]})

    report = audit(classifier, examples, LABELS)

    assert report.max_group_f1_gap == 0.0
    assert report.effectiveness_score == report.overall.f1_macro


def test_audit_rejects_empty_examples():
    with pytest.raises(ValueError):
        audit(DictClassifier({}), [], LABELS)

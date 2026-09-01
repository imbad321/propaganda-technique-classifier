import json

import numpy as np

import evaluate
from labels import LABELS


def test_load_jsonl_reads_each_line(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"text": "a"}\n{"text": "b"}\n', encoding="utf-8")
    assert evaluate.load_jsonl(path) == [{"text": "a"}, {"text": "b"}]


def test_load_jsonl_missing_file_returns_empty_list(tmp_path):
    assert evaluate.load_jsonl(tmp_path / "missing.jsonl") == []


def test_load_thresholds_defaults_when_file_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate, "MODEL_DIR", tmp_path)
    assert (evaluate.load_thresholds() == evaluate.THRESHOLD).all()


def test_label_confusion_counts_cross_label_pairs_only():
    # row 0: true=loaded_language, pred=loaded_language+name_calling -> one cross pair
    # row 1: true=pred=name_calling -> no cross pair
    y_true = np.array([[1, 0], [0, 1]])
    y_pred = np.array([[1, 1], [0, 1]])
    confusion = evaluate.label_confusion(y_true, y_pred)
    assert confusion == {LABELS[0]: {LABELS[1]: 1}}

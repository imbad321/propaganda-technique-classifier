import json

import inference
from labels import LABELS


def test_split_sentences_splits_and_strips():
    text = "The bill passed 218-210.  It takes effect in June."
    assert inference.split_sentences(text) == [
        "The bill passed 218-210.",
        "It takes effect in June.",
    ]


def test_split_sentences_drops_blank_input():
    assert inference.split_sentences("   ") == []


def test_load_thresholds_defaults_when_file_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(inference, "MODEL_DIR", tmp_path)
    assert inference._load_thresholds() == [inference.THRESHOLD] * len(LABELS)


def test_load_thresholds_reads_per_label_overrides(monkeypatch, tmp_path):
    (tmp_path / "thresholds.json").write_text(
        json.dumps({LABELS[0]: 0.7}), encoding="utf-8"
    )
    monkeypatch.setattr(inference, "MODEL_DIR", tmp_path)
    thresholds = inference._load_thresholds()
    assert thresholds[0] == 0.7
    assert thresholds[1] == inference.THRESHOLD

import json

from bias_audit.io import load_labeled_jsonl
from bias_audit.types import LabeledExample


def test_load_labeled_jsonl_default_fields(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps({"text": "hello", "labels": [1, 0], "group": "x"}) + "\n",
        encoding="utf-8",
    )
    examples = load_labeled_jsonl(path)
    assert examples == [LabeledExample(text="hello", labels=[1, 0], group="x")]


def test_load_labeled_jsonl_custom_field_names(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps({"sentence": "hi", "y": [0, 1], "lean": "left"}) + "\n",
        encoding="utf-8",
    )
    examples = load_labeled_jsonl(
        path, text_field="sentence", labels_field="y", group_field="lean"
    )
    assert examples == [LabeledExample(text="hi", labels=[0, 1], group="left")]


def test_load_labeled_jsonl_missing_file_returns_empty_list(tmp_path):
    assert load_labeled_jsonl(tmp_path / "missing.jsonl") == []


def test_load_labeled_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        '{"text": "a", "labels": [1], "group": "g"}\n\n', encoding="utf-8"
    )
    assert len(load_labeled_jsonl(path)) == 1

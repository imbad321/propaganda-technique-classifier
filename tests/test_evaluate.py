import evaluate


def test_load_jsonl_reads_each_line(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"text": "a"}\n{"text": "b"}\n', encoding="utf-8")
    assert evaluate.load_jsonl(path) == [{"text": "a"}, {"text": "b"}]


def test_load_jsonl_missing_file_returns_empty_list(tmp_path):
    assert evaluate.load_jsonl(tmp_path / "missing.jsonl") == []


def test_load_thresholds_defaults_when_file_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate, "MODEL_DIR", tmp_path)
    assert (evaluate.load_thresholds() == evaluate.THRESHOLD).all()

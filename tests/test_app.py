import sys
from unittest.mock import patch

import pytest


@pytest.fixture
def client():
    with patch("inference.load_model", return_value=(None, None, None)):
        sys.modules.pop("app", None)
        import app as app_module
    app_module.app.testing = True
    return app_module.app.test_client()


def test_predict_rejects_blank_text(client):
    resp = client.post("/predict", json={"text": "   "})
    assert resp.status_code == 400


def test_predict_rejects_missing_text(client):
    resp = client.post("/predict", json={})
    assert resp.status_code == 400


def test_predict_returns_model_output(client, monkeypatch):
    monkeypatch.setattr(
        "inference.predict",
        lambda text: [{"sentence": text, "labels": []}],
    )
    resp = client.post("/predict", json={"text": "The bill passed 218-210."})
    assert resp.status_code == 200
    assert resp.get_json() == [{"sentence": "The bill passed 218-210.", "labels": []}]

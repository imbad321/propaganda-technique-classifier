import json
from pathlib import Path

import nltk
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from labels import LABELS

MODEL_DIR = Path(__file__).resolve().parent / "model"
THRESHOLD = 0.5
MAX_LENGTH = 128

_model = None
_tokenizer = None
_device = None
_thresholds = None


def _load_thresholds():
    path = MODEL_DIR / "thresholds.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            by_name = json.load(f)
        return [by_name.get(label, THRESHOLD) for label in LABELS]
    return [THRESHOLD] * len(LABELS)


def _ensure_punkt():
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def load_model():
    global _model, _tokenizer, _device, _thresholds
    if _model is not None:
        return _model, _tokenizer, _device

    if not MODEL_DIR.exists() or not any(MODEL_DIR.glob("*.json")):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_DIR}. Run `python train.py` first, "
            "or point MODEL_DIR at a HuggingFace Hub repo id."
        )

    _ensure_punkt()
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    _model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(_device)
    _model.eval()
    _thresholds = _load_thresholds()
    return _model, _tokenizer, _device


def split_sentences(text):
    _ensure_punkt()
    return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]


def predict(text, threshold=None):
    model, tokenizer, device = load_model()
    sentences = split_sentences(text)
    if not sentences:
        return []

    thresholds = [threshold] * len(LABELS) if threshold is not None else _thresholds

    with torch.no_grad():
        enc = tokenizer(sentences, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt").to(device)
        logits = model(**enc).logits
        probs = torch.sigmoid(logits).cpu().numpy()

    results = []
    for sentence, prob_row in zip(sentences, probs):
        active = [
            {"name": LABELS[i], "score": round(float(score), 4)}
            for i, score in enumerate(prob_row)
            if score >= thresholds[i]
        ]
        active.sort(key=lambda x: x["score"], reverse=True)
        results.append({"sentence": sentence, "labels": active})
    return results

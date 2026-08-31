import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from labels import LABELS

MODEL_DIR = Path(__file__).resolve().parent / "model"
VAL_PATH = Path(__file__).resolve().parent / "data" / "processed" / "val.jsonl"
MAX_LENGTH = 128
BATCH_SIZE = 32
CANDIDATE_THRESHOLDS = np.arange(0.05, 0.96, 0.05)


def main():
    with open(VAL_PATH, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    texts = [r["text"] for r in rows]
    y_true = np.array([r["labels"] for r in rows])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(device)
    model.eval()

    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            enc = tokenizer(batch, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs.append(torch.sigmoid(logits).cpu().numpy())
    probs = np.concatenate(probs, axis=0)

    thresholds = {}
    print(f"{'label':<28}{'default(0.5) F1':>17}{'best_thr':>10}{'best F1':>10}")
    for i, label in enumerate(LABELS):
        best_thr, best_f1 = 0.5, f1_score(y_true[:, i], (probs[:, i] >= 0.5).astype(int), zero_division=0)
        default_f1 = best_f1
        for thr in CANDIDATE_THRESHOLDS:
            preds = (probs[:, i] >= thr).astype(int)
            f1 = f1_score(y_true[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_thr, best_f1 = float(thr), f1
        thresholds[label] = round(best_thr, 2)
        print(f"{label:<28}{default_f1:>17.3f}{best_thr:>10.2f}{best_f1:>10.3f}")

    macro_default = f1_score(y_true, (probs >= 0.5).astype(int), average="macro", zero_division=0)
    tuned_preds = np.zeros_like(y_true)
    for i, label in enumerate(LABELS):
        tuned_preds[:, i] = (probs[:, i] >= thresholds[label]).astype(int)
    macro_tuned = f1_score(y_true, tuned_preds, average="macro", zero_division=0)
    print(f"\nValidation macro F1: default=0.5 -> {macro_default:.3f}, tuned -> {macro_tuned:.3f}")

    with open(MODEL_DIR / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)
    print(f"Wrote {MODEL_DIR / 'thresholds.json'}")


if __name__ == "__main__":
    main()

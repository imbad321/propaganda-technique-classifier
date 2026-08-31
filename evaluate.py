import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from labels import LABELS, NUM_LABELS

MODEL_DIR = Path(__file__).resolve().parent / "model"
TEST_PATH = Path(__file__).resolve().parent / "data" / "processed" / "test.jsonl"
BIAS_CHECK_PATH = Path(__file__).resolve().parent / "data" / "bias_check" / "bias_check_set.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
THRESHOLD = 0.5
MAX_LENGTH = 128
BATCH_SIZE = 32


def load_jsonl(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_thresholds():
    path = MODEL_DIR / "thresholds.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            by_name = json.load(f)
        return np.array([by_name.get(label, THRESHOLD) for label in LABELS])
    return np.array([THRESHOLD] * len(LABELS))


def predict_probs(model, tokenizer, texts, device):
    all_probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            enc = tokenizer(batch, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, NUM_LABELS))


def label_confusion(y_true, y_pred):
    confusion = defaultdict(lambda: defaultdict(int))
    for true_row, pred_row in zip(y_true, y_pred):
        true_labels = [LABELS[i] for i, v in enumerate(true_row) if v == 1]
        pred_labels = [LABELS[i] for i, v in enumerate(pred_row) if v == 1]
        for t in true_labels:
            for p in pred_labels:
                if t != p:
                    confusion[t][p] += 1
    return {t: dict(p) for t, p in confusion.items()}


def evaluate_split(model, tokenizer, device, examples, thresholds, label_name="test"):
    texts = [ex["text"] for ex in examples]
    y_true = np.array([ex["labels"] for ex in examples])
    probs = predict_probs(model, tokenizer, texts, device)
    y_pred = (probs >= thresholds).astype(int)

    result = {
        "n_examples": len(examples),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "per_label_f1": {
            label: f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
            for i, label in enumerate(LABELS)
        },
        "confusion": label_confusion(y_true, y_pred),
    }
    print(f"\n=== {label_name} ===")
    print(f"n={result['n_examples']}  F1_micro={result['f1_micro']:.3f}  F1_macro={result['f1_macro']:.3f}")
    for label, f1 in result["per_label_f1"].items():
        print(f"  {label}: {f1:.3f}")
    return result, y_pred


def evaluate_bias_check(model, tokenizer, device, examples, thresholds):
    by_lean = defaultdict(list)
    for ex in examples:
        by_lean[ex["lean"]].append(ex)

    overall, _ = evaluate_split(model, tokenizer, device, examples, thresholds, label_name="bias-check (overall)")
    per_lean = {}
    for lean, lean_examples in sorted(by_lean.items()):
        result, _ = evaluate_split(model, tokenizer, device, lean_examples, thresholds, label_name=f"bias-check (lean={lean})")
        per_lean[lean] = result

    macro_f1s = [r["f1_macro"] for r in per_lean.values()]
    spread = (max(macro_f1s) - min(macro_f1s)) if macro_f1s else None
    effectiveness_score = (overall["f1_macro"] - spread) if spread is not None else None

    print(f"\nMax macro-F1 gap across lean buckets: {spread:.3f}" if spread is not None else "")
    if effectiveness_score is not None:
        print(f"Effectiveness score (bias-check F1 macro - max lean gap): {effectiveness_score:.3f}")

    return {
        "overall": overall,
        "per_lean": per_lean,
        "max_lean_f1_gap": spread,
        "effectiveness_score": effectiveness_score,
    }


def main():
    if not MODEL_DIR.exists() or not any(MODEL_DIR.glob("*.json")):
        raise FileNotFoundError(f"No trained model found at {MODEL_DIR}. Run `python train.py` first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    thresholds = load_thresholds()
    if (thresholds != THRESHOLD).any():
        print(f"Using per-label tuned thresholds: {dict(zip(LABELS, thresholds.tolist()))}")

    test_examples = load_jsonl(TEST_PATH)
    if test_examples:
        test_result, _ = evaluate_split(model, tokenizer, device, test_examples, thresholds, label_name="held-out test split")
        with open(RESULTS_DIR / "eval_results.json", "w", encoding="utf-8") as f:
            json.dump(test_result, f, indent=2)
    else:
        print(f"No test split found at {TEST_PATH} - skipping. Run data/prepare_semeval.py first.")

    bias_examples = load_jsonl(BIAS_CHECK_PATH)
    if bias_examples:
        bias_result = evaluate_bias_check(model, tokenizer, device, bias_examples, thresholds)
        with open(RESULTS_DIR / "bias_check_results.json", "w", encoding="utf-8") as f:
            json.dump(bias_result, f, indent=2)
    else:
        print(
            f"\nNo bias-check set found at {BIAS_CHECK_PATH} - skipping. "
            "Run `python data/build_bias_check_set.py prepare-blind`, hand-label the sheet, "
            "then `python data/build_bias_check_set.py merge`."
        )


if __name__ == "__main__":
    main()

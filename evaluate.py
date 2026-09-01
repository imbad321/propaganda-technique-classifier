import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bias_audit import audit, evaluate_predictions, load_labeled_jsonl
from labels import LABELS

MODEL_DIR = Path(__file__).resolve().parent / "model"
TEST_PATH = Path(__file__).resolve().parent / "data" / "processed" / "test.jsonl"
BIAS_CHECK_PATH = Path(__file__).resolve().parent / "data" / "bias_check" / "bias_check_set.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
THRESHOLD = 0.5
MAX_LENGTH = 128
BATCH_SIZE = 32


class HFClassifier:
    """Adapts a HuggingFace multi-label sequence classifier to bias_audit's
    Classifier protocol: predict_proba(texts) -> (n_texts, n_labels) probability array."""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def predict_proba(self, texts):
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(texts), BATCH_SIZE):
                batch = list(texts[i:i + BATCH_SIZE])
                enc = self.tokenizer(
                    batch, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt"
                ).to(self.device)
                logits = self.model(**enc).logits
                all_probs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, len(LABELS)))


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


def _print_split_result(result, label_name):
    print(f"\n=== {label_name} ===")
    print(f"n={result.n_examples}  F1_micro={result.f1_micro:.3f}  F1_macro={result.f1_macro:.3f}")
    for label, f1 in result.per_label_f1.items():
        print(f"  {label}: {f1:.3f}")


def evaluate_test_split(classifier, thresholds):
    """Held-out SemEval test split has no group/lean field - just overall F1."""
    test_rows = load_jsonl(TEST_PATH)
    if not test_rows:
        print(f"No test split found at {TEST_PATH} - skipping. Run data/prepare_semeval.py first.")
        return None

    texts = [row["text"] for row in test_rows]
    y_true = np.array([row["labels"] for row in test_rows])
    probs = classifier.predict_proba(texts)
    y_pred = (probs >= thresholds).astype(int)
    result = evaluate_predictions(y_true, y_pred, LABELS)
    _print_split_result(result, "held-out test split")
    return result


def evaluate_bias_check(classifier, thresholds):
    """Bias-check set has a `lean` group field - this is the bias_audit part:
    overall F1, per-lean F1 breakdown, and the gap-penalized effectiveness score."""
    examples = load_labeled_jsonl(BIAS_CHECK_PATH, group_field="lean")
    if not examples:
        print(
            f"\nNo bias-check set found at {BIAS_CHECK_PATH} - skipping. "
            "Run `python data/build_bias_check_set.py prepare-blind`, hand-label the sheet, "
            "then `python data/build_bias_check_set.py merge`."
        )
        return None

    report = audit(classifier, examples, LABELS, thresholds=thresholds)
    _print_split_result(report.overall, "bias-check (overall)")
    for lean, result in sorted(report.per_group.items()):
        _print_split_result(result, f"bias-check (lean={lean})")
    print(f"\nMax macro-F1 gap across lean buckets: {report.max_group_f1_gap:.3f}")
    print(f"Effectiveness score (bias-check F1 macro - max lean gap): {report.effectiveness_score:.3f}")
    return report


def main():
    if not MODEL_DIR.exists() or not any(MODEL_DIR.glob("*.json")):
        raise FileNotFoundError(f"No trained model found at {MODEL_DIR}. Run `python train.py` first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(device)
    classifier = HFClassifier(model, tokenizer, device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    thresholds = load_thresholds()
    if (thresholds != THRESHOLD).any():
        print(f"Using per-label tuned thresholds: {dict(zip(LABELS, thresholds.tolist()))}")

    test_result = evaluate_test_split(classifier, thresholds)
    if test_result is not None:
        with open(RESULTS_DIR / "eval_results.json", "w", encoding="utf-8") as f:
            json.dump(asdict(test_result), f, indent=2)

    bias_report = evaluate_bias_check(classifier, thresholds)
    if bias_report is not None:
        # Keeps the on-disk shape ("per_lean", "max_lean_f1_gap") that make_plots.py and
        # the historical results/*.json files already use, instead of bias_audit's
        # generic "group" naming.
        bias_result_json = {
            "overall": asdict(bias_report.overall),
            "per_lean": {lean: asdict(result) for lean, result in bias_report.per_group.items()},
            "max_lean_f1_gap": bias_report.max_group_f1_gap,
            "effectiveness_score": bias_report.effectiveness_score,
        }
        with open(RESULTS_DIR / "bias_check_results.json", "w", encoding="utf-8") as f:
            json.dump(bias_result_json, f, indent=2)


if __name__ == "__main__":
    main()

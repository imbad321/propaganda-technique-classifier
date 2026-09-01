"""
Usage (from project root):
    python data/bias_check/agreement.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from bias_audit import label_reliability  # noqa: E402
from labels import LABELS  # noqa: E402

BIAS_DIR = Path(__file__).resolve().parent


def main():
    with open(BIAS_DIR / "bias_check_set.jsonl", encoding="utf-8") as f:
        user_rows = [json.loads(line) for line in f]
    with open(BIAS_DIR / "claude_labels.json", encoding="utf-8") as f:
        claude_by_index = json.load(f)["labels"]

    expected_keys = {f"{i:03d}" for i in range(len(user_rows))}
    missing = expected_keys - set(claude_by_index.keys())
    assert not missing, f"claude_labels.json is missing indices: {sorted(missing)}"

    y_user = np.array([r["labels"] for r in user_rows])
    y_claude = np.array([claude_by_index[f"{i:03d}"] for i in range(len(user_rows))])

    result = label_reliability(y_user, y_claude, LABELS)
    print(f"n={len(user_rows)} sentences\n")
    print(result.summary())


if __name__ == "__main__":
    main()

"""
Usage (from project root):
    python data/bias_check/agreement.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
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

    print(f"n={len(user_rows)} sentences\n")
    print(f"{'label':<28}{'kappa':>8}{'agree%':>9}{'user_pos':>10}{'claude_pos':>11}")
    kappas = []
    for i, label in enumerate(LABELS):
        u, c = y_user[:, i], y_claude[:, i]
        agree_pct = (u == c).mean() * 100
        if u.std() == 0 and c.std() == 0:
            kappa = 1.0 if (u == c).all() else 0.0
        else:
            kappa = cohen_kappa_score(u, c)
        kappas.append(kappa)
        print(f"{label:<28}{kappa:>8.3f}{agree_pct:>8.1f}%{u.sum():>10}{c.sum():>11}")

    print(f"\nMean per-label kappa: {np.mean(kappas):.3f}")

    exact_match = (y_user == y_claude).all(axis=1).mean() * 100
    print(f"Exact full-vector agreement (all 6 labels match): {exact_match:.1f}%")

    any_overlap = ((y_user + y_claude) > 0).sum(axis=1)
    both = ((y_user == 1) & (y_claude == 1)).sum(axis=1)
    jaccard = np.divide(both, any_overlap, out=np.ones_like(both, dtype=float), where=any_overlap != 0)
    print(f"Mean per-sentence Jaccard overlap (active labels only): {jaccard.mean():.3f}")


if __name__ == "__main__":
    main()

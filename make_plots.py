import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR = Path(__file__).resolve().parent / "docs" / "images"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

CONFIGS = [
    ("8ep\nunweighted", "eval_results_8epoch.json", "bias_check_results_8epoch.json", "distilbert"),
    ("20ep\nunweighted", "eval_results_20epoch.json", "bias_check_results_20epoch.json", "distilbert"),
    ("8ep\nweighted", "eval_results_weighted8epoch.json", "bias_check_results_weighted8epoch.json", "distilbert"),
    ("8ep weighted\n+thresholds", "eval_results_weighted8epoch_tunedthresh.json", "bias_check_results_weighted8epoch_tunedthresh.json", "distilbert"),
    ("roberta 8ep\nweighted", "eval_results_roberta_weighted8epoch.json", "bias_check_results_roberta_weighted8epoch.json", "roberta-base"),
    ("8ep weighted\n+BASIL", "eval_results_weighted8epoch_basil.json", "bias_check_results_weighted8epoch_basil.json", "distilbert"),
    ("roberta 8ep\nweighted+BASIL", "eval_results_roberta_weighted8epoch_basil.json", "bias_check_results_roberta_weighted8epoch_basil.json", "roberta-base"),
]


def load():
    rows = []
    for name, eval_f, bias_f, tag in CONFIGS:
        ev = json.load(open(RESULTS_DIR / eval_f))
        bc = json.load(open(RESULTS_DIR / bias_f))
        rows.append({
            "name": name,
            "tag": tag,
            "test_f1": ev["f1_macro"],
            "bias_f1": bc["overall"]["f1_macro"],
            "center": bc["per_lean"]["center"]["f1_macro"],
            "left": bc["per_lean"]["left"]["f1_macro"],
            "right": bc["per_lean"]["right"]["f1_macro"],
            "gap": bc["max_lean_f1_gap"],
        })
    for r in rows:
        r["score"] = r["bias_f1"] - r["gap"]
    return rows


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="both", colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def grouped_bar(rows, keys, colors, labels, title, subtitle, out_path, ylim=0.56):
    x = np.arange(len(rows))
    n = len(keys)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
    for i, (key, color, label) in enumerate(zip(keys, colors, labels)):
        vals = [r[key] for r in rows]
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width * 0.92, color=color, label=label, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([r["name"] for r in rows], fontsize=8.5)
    ax.set_ylim(0, ylim)
    style_axes(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=9, ncols=n)
    fig.suptitle(title, fontsize=13, fontweight="bold", x=0.01, ha="left", color=INK)
    ax.set_title(subtitle, fontsize=9.5, color=MUTED, loc="left", pad=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


def score_bar(rows, out_path):
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    x = np.arange(len(rows))
    scores = [r["score"] for r in rows]
    best_i = int(np.argmax(scores))
    colors = [AQUA if i == best_i else BLUE for i in range(len(rows))]
    bars = ax.bar(x, scores, 0.55, color=colors, zorder=3)
    for b, v in zip(bars, scores):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([r["name"] for r in rows], fontsize=8.5)
    ax.set_ylim(0, 0.24)
    style_axes(ax)
    fig.suptitle("Effectiveness score", fontsize=13, fontweight="bold", x=0.01, ha="left", color=INK)
    ax.set_title("bias-check macro F1 minus the max gap between lean buckets - higher is better, best run highlighted", fontsize=9.5, color=MUTED, loc="left", pad=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load()

    grouped_bar(
        rows, ["test_f1", "bias_f1"], [BLUE, ORANGE],
        ["Test split (in-distribution)", "Bias-check (real-world)"],
        "Overall macro F1: in-distribution vs. real-world",
        "SemEval held-out test split (n=1400) vs. the hand-labeled bias-check set (n=150)",
        OUT_DIR / "overall_f1.png",
    )

    grouped_bar(
        rows, ["center", "left", "right"], [BLUE, ORANGE, AQUA],
        ["Center", "Left", "Right"],
        "Bias-check macro F1 by outlet-lean bucket",
        "50 sentences per bucket, labeled blind to source",
        OUT_DIR / "per_lean_f1.png",
    )

    score_bar(rows, OUT_DIR / "effectiveness_score.png")


if __name__ == "__main__":
    main()

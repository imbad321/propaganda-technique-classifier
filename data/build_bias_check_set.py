"""
Usage:
    python data/build_bias_check_set.py prepare-blind
    # hand-label data/bias_check/blind_sheet.csv
    python data/build_bias_check_set.py merge
"""

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from labels import LABELS  # noqa: E402

BIAS_DIR = Path(__file__).resolve().parent / "bias_check"
RAW_CSV = BIAS_DIR / "raw_sentences.csv"
BLIND_SHEET = BIAS_DIR / "blind_sheet.csv"
HIDDEN_MAPPING = BIAS_DIR / "hidden_mapping.json"
OUTPUT_JSONL = BIAS_DIR / "bias_check_set.jsonl"

RANDOM_SEED = 7
TARGET_SIZE_HINT = (150, 200)


def prepare_blind():
    if not RAW_CSV.exists():
        BIAS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RAW_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sentence", "outlet", "lean"])
        print(
            f"No raw sentences found - created an empty template at {RAW_CSV}.\n"
            f"Fill it in with {TARGET_SIZE_HINT[0]}-{TARGET_SIZE_HINT[1]} sentences pulled evenly "
            "across outlets spanning different political leans (columns: sentence, outlet, lean), "
            "then re-run this command."
        )
        return

    with open(RAW_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"{RAW_CSV} is empty. Add sentences before running prepare-blind.")
        return

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(rows)

    hidden_mapping = {}
    with open(BLIND_SHEET, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "sentence"] + LABELS)
        for i, row in enumerate(rows):
            row_id = f"bc{i:04d}"
            hidden_mapping[row_id] = {"outlet": row["outlet"], "lean": row["lean"]}
            writer.writerow([row_id, row["sentence"]] + [""] * len(LABELS))

    with open(HIDDEN_MAPPING, "w", encoding="utf-8") as f:
        json.dump(hidden_mapping, f, indent=2)

    lean_counts = {}
    for meta in hidden_mapping.values():
        lean_counts[meta["lean"]] = lean_counts.get(meta["lean"], 0) + 1

    print(f"Wrote {len(rows)} shuffled, source-stripped rows to {BLIND_SHEET}.")
    print(f"Lean distribution (kept hidden from the labeling sheet): {lean_counts}")
    print(
        "Now hand-label blind_sheet.csv: put a 1 in each label column that applies to the "
        "sentence (0 or blank otherwise), using labels.LABEL_DESCRIPTIONS as your rubric. "
        "You will not see the outlet or lean while doing this - that's the point."
    )


def merge():
    if not BLIND_SHEET.exists() or not HIDDEN_MAPPING.exists():
        print("Run `prepare-blind` first, then hand-label blind_sheet.csv before merging.")
        return

    with open(HIDDEN_MAPPING, encoding="utf-8") as f:
        hidden_mapping = json.load(f)

    examples = []
    with open(BLIND_SHEET, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row_id = row["id"]
            if row_id not in hidden_mapping:
                continue
            try:
                vec = [int(row[label]) if row[label].strip() else 0 for label in LABELS]
            except ValueError:
                raise ValueError(f"Row {row_id} has a non-0/1 value in a label column - fix it and re-run.")
            meta = hidden_mapping[row_id]
            examples.append({
                "text": row["sentence"],
                "labels": vec,
                "outlet": meta["outlet"],
                "lean": meta["lean"],
            })

    if not examples:
        print("No labeled rows found - nothing to merge.")
        return

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(examples)} labeled examples to {OUTPUT_JSONL}.")
    print("This file is read directly by evaluate.py for the per-lean F1 breakdown.")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("prepare-blind", "merge"):
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] == "prepare-blind":
        prepare_blind()
    else:
        merge()


if __name__ == "__main__":
    main()

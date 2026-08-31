# BASIL repo (github.com/launchnlp/BASIL) ships no LICENSE file - local
# research use only, don't publish data/raw/BASIL or data/processed/basil.jsonl.

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from labels import LABELS, LABEL2ID  # noqa: E402

REPO_URL = "https://github.com/launchnlp/BASIL.git"
RAW_DIR = Path(__file__).resolve().parent / "raw" / "BASIL"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "basil.jsonl"

SOURCE_TO_LEAN = {"nyt": "center", "fox": "right", "hpo": "left"}


def clone_if_needed():
    if RAW_DIR.exists():
        print(f"BASIL already cloned at {RAW_DIR}, skipping.")
        return
    print(f"Cloning {REPO_URL} ...")
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(RAW_DIR)], check=True)


def flatten_sentences(body_paragraphs):
    """body-paragraphs is a list of paragraphs, each already a list of
    sentence strings - just flatten, no re-splitting needed."""
    return [s for para in body_paragraphs for s in para]


def labels_for_sentence(sentence, spans):
    active = set()
    for span in spans:
        if span["txt"] not in sentence:
            continue
        if span["bias"] == "lex":
            if span["quote"] == "yes":
                continue  # quoting someone else's word choice, not the outlet's
            active.add("loaded_language")
        elif span["bias"] == "inf":
            active.add("unsupported_claim")
    if not active:
        active.add("factual_neutral")
    vec = [0] * len(LABELS)
    for label in active:
        vec[LABEL2ID[label]] = 1
    return vec


def build_examples():
    examples = []
    article_paths = sorted(RAW_DIR.glob("articles/*/*.json"))
    for article_path in article_paths:
        year_dir, filename = article_path.parent.name, article_path.stem
        ann_path = RAW_DIR / "annotations" / year_dir / f"{filename}_ann.json"
        if not ann_path.exists():
            continue

        article = json.loads(article_path.read_text(encoding="utf-8"))
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        source = article["source"]
        lean = SOURCE_TO_LEAN.get(source)
        if lean is None:
            continue

        sentences = flatten_sentences(article["body-paragraphs"])
        spans = ann.get("phrase-level-annotations", [])

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 3:
                continue
            vec = labels_for_sentence(sentence, spans)
            examples.append({
                "text": sentence,
                "labels": vec,
                "source": "basil",
                "outlet": source,
                "lean": lean,
                "article_uuid": article["uuid"],
            })
    return examples


def main():
    clone_if_needed()
    examples = build_examples()

    label_counts = {label: 0 for label in LABELS}
    for ex in examples:
        for i, v in enumerate(ex["labels"]):
            if v:
                label_counts[LABELS[i]] += 1

    print(f"Built {len(examples)} sentence-level examples from BASIL.")
    print("Label distribution (sentences can have multiple labels):")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {OUT_PATH}")
    print(
        "Not merged into train.jsonl automatically - pass "
        f"`--extra-train {OUT_PATH}` to train.py to include it."
    )


if __name__ == "__main__":
    main()

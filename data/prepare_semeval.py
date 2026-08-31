import json
import os
import random
import sys
import tarfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from labels import LABELS, LABEL2ID, TECHNIQUE_TO_LABEL  # noqa: E402

DATA_URL = "https://zenodo.org/records/3952415/files/datasets-v2.tgz?download=1"
RAW_DIR = Path(__file__).resolve().parent / "raw"
ARCHIVE_PATH = RAW_DIR / "datasets-v2.tgz"
PROCESSED_DIR = Path(__file__).resolve().parent / "processed"

TRAIN_FRAC, VAL_FRAC = 0.8, 0.1  # remainder (0.1) goes to test
RANDOM_SEED = 13


def download_archive():
    if ARCHIVE_PATH.exists():
        print(f"Archive already present at {ARCHIVE_PATH}, skipping download.")
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATA_URL} ...")
    resp = requests.get(DATA_URL, stream=True, timeout=60)
    resp.raise_for_status()
    with open(ARCHIVE_PATH, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    print(f"Saved to {ARCHIVE_PATH}")


def extract_archive():
    # The archive's top-level folder is "datasets" (not "datasets-v2").
    extract_dir = RAW_DIR / "datasets"
    if extract_dir.exists():
        print(f"Already extracted at {extract_dir}, skipping.")
        return extract_dir
    print("Extracting archive...")
    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        tar.extractall(RAW_DIR)
    if not extract_dir.exists():
        # Fall back to RAW_DIR in case a future archive revision changes the
        # top-level folder name again.
        return RAW_DIR
    return extract_dir


def find_dirs(root):
    """The archive layout has shifted slightly between shared-task years, so
    search for the article/label directories by name pattern instead of a
    single hardcoded path. As of the Zenodo v2 mirror: datasets/train-articles
    and datasets/train-labels-task2-technique-classification."""
    articles_dirs = sorted(d for d in root.glob("**/train-articles") if d.is_dir())
    labels_dirs = sorted(
        d for d in root.glob("**/*") if d.is_dir() and "train-labels" in d.name and (
            "tc" in d.name.lower() or "technique" in d.name.lower()
        )
    )
    if not articles_dirs or not labels_dirs:
        raise FileNotFoundError(
            f"Could not locate train-articles / train-labels-task-tc directories under {root}. "
            "The archive layout may have changed - inspect it manually and adjust find_dirs()."
        )
    return articles_dirs[0], labels_dirs[0]


def parse_tc_labels(path):
    """Each line: article_id \\t technique_name \\t start_offset \\t end_offset"""
    spans = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            _, technique, start, end = parts
            spans.append((technique, int(start), int(end)))
    return spans


def sentence_split_with_offsets(text):
    """nltk.sent_tokenize doesn't return character offsets, so we recover them
    by scanning the source text left-to-right for each returned sentence."""
    import nltk

    sentences = nltk.sent_tokenize(text)
    offsets = []
    cursor = 0
    for sent in sentences:
        start = text.find(sent, cursor)
        if start == -1:
            # Whitespace/quote normalization can occasionally break the match;
            # skip rather than mis-align offsets.
            continue
        end = start + len(sent)
        offsets.append((sent, start, end))
        cursor = end
    return offsets


def ensure_nltk_punkt():
    import nltk

    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def build_examples(articles_dir, labels_dir):
    examples = []
    label_counts = {label: 0 for label in LABELS}
    article_ids = []

    for article_path in sorted(articles_dir.glob("article*.txt")):
        article_id = article_path.stem.replace("article", "")
        candidates = list(labels_dir.glob(f"article{article_id}.*labels"))
        if not candidates:
            continue
        label_path = candidates[0]

        text = article_path.read_text(encoding="utf-8")
        spans = parse_tc_labels(label_path)
        sentences = sentence_split_with_offsets(text)
        if not sentences:
            continue
        article_ids.append(article_id)

        for sent, start, end in sentences:
            sent = sent.strip()
            if len(sent) < 3:
                continue
            active = set()
            for technique, span_start, span_end in spans:
                if span_start < end and span_end > start:  # overlap
                    mapped = TECHNIQUE_TO_LABEL.get(technique)
                    if mapped:
                        active.add(mapped)
            if not active:
                active.add("factual_neutral")

            vec = [0] * len(LABELS)
            for label in active:
                vec[LABEL2ID[label]] = 1
                label_counts[label] += 1

            examples.append({"text": sent, "labels": vec, "article_id": article_id})

    return examples, label_counts, article_ids


def split_by_article(examples, article_ids):
    rng = random.Random(RANDOM_SEED)
    ids = list(article_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train + n_val])
    test_ids = set(ids[n_train + n_val:])

    splits = {"train": [], "val": [], "test": []}
    for ex in examples:
        if ex["article_id"] in train_ids:
            splits["train"].append(ex)
        elif ex["article_id"] in val_ids:
            splits["val"].append(ex)
        else:
            splits["test"].append(ex)
    return splits


def write_jsonl(path, examples):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def main():
    ensure_nltk_punkt()
    download_archive()
    extract_dir = extract_archive()
    articles_dir, labels_dir = find_dirs(extract_dir)
    print(f"Using articles dir: {articles_dir}")
    print(f"Using labels dir:   {labels_dir}")

    examples, label_counts, article_ids = build_examples(articles_dir, labels_dir)
    print(f"Built {len(examples)} sentence-level examples from {len(article_ids)} articles.")
    print("Label distribution (sentences can have multiple labels):")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")

    splits = split_by_article(examples, article_ids)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, split_examples in splits.items():
        out_path = PROCESSED_DIR / f"{split_name}.jsonl"
        write_jsonl(out_path, split_examples)
        print(f"Wrote {len(split_examples)} examples to {out_path}")

    with open(PROCESSED_DIR / "label_counts.json", "w", encoding="utf-8") as f:
        json.dump(label_counts, f, indent=2)


if __name__ == "__main__":
    main()

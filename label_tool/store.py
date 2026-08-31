import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path

BIAS_DIR = Path(__file__).resolve().parent.parent / "data" / "bias_check"
ENTRIES_PATH = BIAS_DIR / "entries.json"
EXPORT_PATH = BIAS_DIR / "bias_check_set.jsonl"


def _now():
    return datetime.now(timezone.utc).isoformat()


def load_entries():
    if not ENTRIES_PATH.exists():
        return []
    with open(ENTRIES_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_entries(entries):
    BIAS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ENTRIES_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def _next_id(entries):
    n = len(entries)
    while True:
        candidate = f"e{n:04d}"
        if not any(e["id"] == candidate for e in entries):
            return candidate
        n += 1


def add_entry(sentence, outlet, lean):
    sentence = sentence.strip()
    if not sentence:
        return None, "Sentence is empty."
    entries = load_entries()
    entry = {
        "id": _next_id(entries),
        "sentence": sentence,
        "outlet": outlet.strip(),
        "lean": lean.strip(),
        "labels": None,
        "labeled": False,
        "created_at": _now(),
    }
    entries.append(entry)
    save_entries(entries)
    return entry, None


def bulk_import_csv(text):
    entries = load_entries()
    added = 0
    errors = []
    reader = csv.reader(io.StringIO(text))
    for i, row in enumerate(reader, start=1):
        row = [c.strip() for c in row]
        if not row or not any(row):
            continue
        if row[0].lower() == "sentence" and len(row) >= 2 and row[1].lower() == "outlet":
            continue  # skip an optional header row
        if len(row) != 3:
            errors.append(f"Line {i}: expected 3 columns (sentence,outlet,lean), got {len(row)}.")
            continue
        sentence, outlet, lean = row
        if not sentence:
            errors.append(f"Line {i}: empty sentence.")
            continue
        entries.append({
            "id": _next_id(entries),
            "sentence": sentence,
            "outlet": outlet,
            "lean": lean,
            "labels": None,
            "labeled": False,
            "created_at": _now(),
        })
        added += 1
    save_entries(entries)
    return added, errors


def get_by_id(entry_id):
    for e in load_entries():
        if e["id"] == entry_id:
            return e
    return None


def get_next_unlabeled(exclude_id=None):
    entries = [e for e in load_entries() if not e["labeled"] and e["id"] != exclude_id]
    return entries[0] if entries else None


def set_labels(entry_id, label_vector):
    entries = load_entries()
    for e in entries:
        if e["id"] == entry_id:
            e["labels"] = label_vector
            e["labeled"] = True
            save_entries(entries)
            return e
    return None


def delete_entry(entry_id):
    entries = load_entries()
    remaining = [e for e in entries if e["id"] != entry_id]
    if len(remaining) == len(entries):
        return False
    save_entries(remaining)
    return True


def stats():
    entries = load_entries()
    total = len(entries)
    labeled = sum(1 for e in entries if e["labeled"])
    by_lean = {}
    for e in entries:
        lean = e["lean"] or "(unspecified)"
        by_lean.setdefault(lean, {"total": 0, "labeled": 0})
        by_lean[lean]["total"] += 1
        if e["labeled"]:
            by_lean[lean]["labeled"] += 1
    return {"total": total, "labeled": labeled, "by_lean": by_lean}


def export_labeled():
    entries = [e for e in load_entries() if e["labeled"]]
    BIAS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EXPORT_PATH, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps({
                "text": e["sentence"],
                "labels": e["labels"],
                "outlet": e["outlet"],
                "lean": e["lean"],
            }) + "\n")
    by_lean = {}
    for e in entries:
        lean = e["lean"] or "(unspecified)"
        by_lean[lean] = by_lean.get(lean, 0) + 1
    return {"path": str(EXPORT_PATH), "count": len(entries), "by_lean": by_lean}

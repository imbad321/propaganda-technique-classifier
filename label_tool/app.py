"""
Usage:
    python label_tool/app.py
Visit http://127.0.0.1:5001
"""

import sys
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, url_for

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from labels import LABELS, LABEL_DESCRIPTIONS  # noqa: E402

import store  # noqa: E402

app = Flask(__name__)


@app.route("/")
def index():
    return redirect(url_for("add"))


@app.route("/add", methods=["GET", "POST"])
def add():
    message = None
    error = None
    if request.method == "POST":
        if request.form.get("mode") == "bulk":
            added, errors = store.bulk_import_csv(request.form.get("bulk_text", ""))
            message = f"Imported {added} sentence(s)."
            if errors:
                error = " ".join(errors[:10]) + (" ..." if len(errors) > 10 else "")
        else:
            entry, err = store.add_entry(
                request.form.get("sentence", ""),
                request.form.get("outlet", ""),
                request.form.get("lean", ""),
            )
            if err:
                error = err
            else:
                message = f"Added {entry['id']}."

    s = store.stats()
    leans = sorted(l for l in s["by_lean"].keys() if l != "(unspecified)")
    return render_template("add.html", stats=s, leans=leans, message=message, error=error)


@app.route("/label")
def label_next():
    entry = store.get_next_unlabeled()
    if entry is None:
        return render_template("label.html", entry=None, labels=LABELS, descriptions=LABEL_DESCRIPTIONS, stats=store.stats())
    return redirect(url_for("label_entry", entry_id=entry["id"]))


@app.route("/label/<entry_id>", methods=["GET", "POST"])
def label_entry(entry_id):
    entry = store.get_by_id(entry_id)
    if entry is None:
        return redirect(url_for("label_next"))

    if request.method == "POST":
        vector = [1 if request.form.get(label) == "on" else 0 for label in LABELS]
        store.set_labels(entry_id, vector)
        next_entry = store.get_next_unlabeled(exclude_id=entry_id)
        if next_entry:
            return redirect(url_for("label_entry", entry_id=next_entry["id"]))
        return redirect(url_for("label_next"))

    blind_entry = {"id": entry["id"], "sentence": entry["sentence"], "labels": entry["labels"]}
    return render_template("label.html", entry=blind_entry, labels=LABELS, descriptions=LABEL_DESCRIPTIONS, stats=store.stats())


@app.route("/entries")
def entries():
    return render_template("entries.html", entries=store.load_entries(), labels=LABELS)


@app.route("/entries/<entry_id>/delete", methods=["POST"])
def delete_entry(entry_id):
    store.delete_entry(entry_id)
    return redirect(url_for("entries"))


@app.route("/export", methods=["GET", "POST"])
def export():
    result = None
    if request.method == "POST":
        result = store.export_labeled()
    return render_template("export.html", stats=store.stats(), result=result)


@app.route("/api/stats")
def api_stats():
    return jsonify(store.stats())


if __name__ == "__main__":
    app.run(debug=True, port=5001)

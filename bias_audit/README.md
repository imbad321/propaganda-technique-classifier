# bias-audit

A small library for one specific question: **did my text classifier learn the task, or
did it learn a proxy for the task** - the outlet a sentence came from, the demographic
of its author, the register of the writing - instead of the thing you actually asked it
to detect?

This is a narrower, more practical cousin of ideas from the shortcut-learning /
spurious-correlation literature (e.g. group-robustness work like Sagawa et al.'s
group-DRO, behavioral testing like Ribeiro et al.'s CheckList) and of general fairness
toolkits (Fairlearn, AIF360, Aequitas). Those are broader - built mostly around tabular
data and demographic-parity style fairness metrics. `bias-audit` is aimed squarely at
multi-label text classifiers and the specific failure mode of "the model fingerprinted
the source instead of reading the content," with two things bundled in that general
toolkits don't give you out of the box: a single gameable-resistant **effectiveness
score**, and a **label-reliability check** for your own ground truth.

It was extracted from [`propaganda-technique-classifier`](..), which is its worked
example - see that repo's README for `bias-audit` applied end to end against a real
model, or the [project page](https://imbad321.github.io/propaganda-technique-classifier/bias-audit/)
for this same README in page form.

## Install

```bash
pip install -e ./bias_audit
```

## Usage

```python
from bias_audit import Classifier, LabeledExample, audit, label_reliability

# 1. Wrap your model so it can score raw text against your label set.
class MyClassifier:
    def predict_proba(self, texts):
        ...  # returns an (n_texts, n_labels) array of scores in [0, 1]

# 2. Build a labeled, group-tagged evaluation set. `group` is any categorical
#    split you're worried the model might be keying off of instead of content -
#    outlet lean, dialect, age bracket, whatever applies to your problem.
examples = [
    LabeledExample(text="...", labels=[1, 0, 0], group="left"),
    LabeledExample(text="...", labels=[0, 1, 0], group="right"),
    # ...
]

# 3. Audit it.
report = audit(MyClassifier(), examples, label_names=["a", "b", "c"])
print(report.summary())
print(report.effectiveness_score)   # overall F1_macro minus the max group-to-group gap
print(report.per_group)             # per-group SplitResult breakdown

# 4. Optional: check your own ground truth against a second labeler before
#    trusting the audit above - a low kappa on one label means that label's
#    ground truth is the noisy part, not necessarily the model.
reliability = label_reliability(your_labels, second_labeler_labels, label_names=["a", "b", "c"])
print(reliability.summary())
```

`load_labeled_jsonl(path, text_field=..., labels_field=..., group_field=...)` reads a
JSONL file straight into a list of `LabeledExample`s if that's an easier starting point
than constructing them by hand.

## API

- `LabeledExample(text, labels, group)` - one example.
- `Classifier` - a `Protocol` with one method, `predict_proba(texts) -> np.ndarray`.
  Anything that implements it works: a HuggingFace model, an sklearn pipeline, a bare
  function wrapped in a class.
- `audit(classifier, examples, label_names, thresholds=0.5) -> AuditReport` - the main
  entry point. `AuditReport` has `overall` (a `SplitResult`: n, F1 micro/macro,
  per-label F1, confusion), `per_group` (the same, per group value), `max_group_f1_gap`,
  and `effectiveness_score`.
- `evaluate_predictions(y_true, y_pred, label_names) -> SplitResult` and
  `label_confusion(y_true, y_pred, label_names)` - the lower-level pieces `audit()` is
  built from, if you already have predictions and don't need the `Classifier` wrapper.
- `label_reliability(labels_a, labels_b, label_names) -> LabelReliability` - per-label
  Cohen's kappa, exact-match rate, and mean Jaccard overlap between two independent
  labelings of the same examples.
- `load_labeled_jsonl(path, ...) -> list[LabeledExample]`.

## Tests

```bash
pip install -e ./bias_audit pytest
pytest bias_audit/tests
```

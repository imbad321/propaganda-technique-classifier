from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class LabeledExample:
    """One labeled example: text, a multi-hot label vector, and the group it belongs to.

    `group` is whatever categorical split you want the audit broken down by - outlet
    lean, dialect, gender, age bracket, anything with a handful of discrete values.
    """

    text: str
    labels: Sequence[int]
    group: str


class Classifier(Protocol):
    """Anything that scores texts against a fixed, ordered set of labels.

    `predict_proba` must return an (n_examples, n_labels) array of scores in [0, 1],
    in the same label order as the `labels` vectors on your `LabeledExample`s. Wrap
    whatever model you have - a HuggingFace model, an sklearn pipeline, a bare
    function - in a small adapter class that implements this one method.
    """

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray: ...

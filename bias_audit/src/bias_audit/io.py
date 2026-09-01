import json
from pathlib import Path
from typing import List, Union

from .types import LabeledExample


def load_labeled_jsonl(
    path: Union[str, Path],
    text_field: str = "text",
    labels_field: str = "labels",
    group_field: str = "group",
) -> List[LabeledExample]:
    """Loads a JSONL file of `{text_field: ..., labels_field: [0/1, ...], group_field: ...}`
    rows into `LabeledExample`s. Returns an empty list if `path` doesn't exist, so callers
    can treat a not-yet-built evaluation set as "nothing to audit yet" rather than an error.
    """
    path = Path(path)
    if not path.exists():
        return []
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append(
                LabeledExample(
                    text=row[text_field],
                    labels=row[labels_field],
                    group=row[group_field],
                )
            )
    return examples

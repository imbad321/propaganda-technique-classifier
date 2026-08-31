import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import Sequence, Value, load_dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from labels import LABELS, LABEL2ID, ID2LABEL, NUM_LABELS

PROCESSED_DIR = Path(__file__).resolve().parent / "data" / "processed"
MODEL_OUT_DIR = Path(__file__).resolve().parent / "model"
THRESHOLD = 0.5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="distilbert-base-uncased")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--loss-weighting", choices=["balanced", "none"], default="balanced")
    p.add_argument("--max-pos-weight", type=float, default=30.0)
    p.add_argument(
        "--extra-train", action="append", default=None,
        help="Additional train-only JSONL file(s) to concatenate onto train.jsonl "
             "(e.g. data/processed/basil.jsonl from data/prepare_basil.py). "
             "Only 'text' and 'labels' columns are used - repeat the flag for multiple files.",
    )
    return p.parse_args()


def load_splits(extra_train_paths=None):
    train_path = PROCESSED_DIR / "train.jsonl"
    val_path = PROCESSED_DIR / "val.jsonl"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Missing {train_path} or {val_path}. Run `python data/prepare_semeval.py` first."
        )
    dataset = load_dataset("json", data_files={"train": str(train_path), "validation": str(val_path)})

    if extra_train_paths:
        from datasets import concatenate_datasets

        base_train = dataset["train"].select_columns(["text", "labels"])
        parts = [base_train]
        for path in extra_train_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"--extra-train file not found: {path}")
            extra = load_dataset("json", data_files=str(path))["train"].select_columns(["text", "labels"])
            print(f"Adding {len(extra)} examples from {path}")
            parts.append(extra)
        dataset["train"] = concatenate_datasets(parts)
        print(f"Combined train set: {len(dataset['train'])} examples")

    return dataset


def compute_pos_weight(raw_train_dataset, max_pos_weight):
    arr = np.array(raw_train_dataset["labels"], dtype=np.float64)
    pos = arr.sum(axis=0)
    neg = arr.shape[0] - pos
    pos = np.clip(pos, 1, None)
    weight = np.clip(neg / pos, 1.0, max_pos_weight)
    return torch.tensor(weight, dtype=torch.float32)


class WeightedTrainer(Trainer):
    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(outputs.logits.device))
        loss = loss_fct(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, label_ids = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= THRESHOLD).astype(int)
    return {
        "f1_micro": f1_score(label_ids, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(label_ids, preds, average="macro", zero_division=0),
    }


def main():
    args = parse_args()
    dataset = load_splits(args.extra_train)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        enc = tokenizer(batch["text"], truncation=True, max_length=args.max_length)
        enc["labels"] = [[float(v) for v in row] for row in batch["labels"]]
        return enc

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)
    tokenized = tokenized.cast_column("labels", Sequence(feature=Value("float32")))
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=str(MODEL_OUT_DIR / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        save_total_limit=2,
        report_to=[],
        fp16=torch.cuda.is_available(),
    )

    if args.loss_weighting == "balanced":
        pos_weight = compute_pos_weight(dataset["train"], args.max_pos_weight)
        print(f"Class-weighted loss enabled. pos_weight per label: "
              f"{dict(zip(LABELS, [round(w, 2) for w in pos_weight.tolist()]))}")
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            pos_weight=pos_weight,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )

    trainer.train()
    val_metrics = trainer.evaluate()
    print("Final validation metrics:", val_metrics)

    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_OUT_DIR))
    tokenizer.save_pretrained(str(MODEL_OUT_DIR))
    with open(MODEL_OUT_DIR / "val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)
    print(f"Saved fine-tuned model + tokenizer to {MODEL_OUT_DIR}")


if __name__ == "__main__":
    main()

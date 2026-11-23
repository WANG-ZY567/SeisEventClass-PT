#!/usr/bin/env python3
"""Strict, transparent shared-label (earthquake/explosion) evaluation.

This tool evaluates a six-class EVT6 model on a ComCat/PNW test set whose
ground-truth labels are restricted to ``eq`` (0) and ``ep`` (1).  It reports
two deliberately distinct protocols:

* ``shared_label_projection``: a closed shared-label decision, obtained by
  comparing only the model probabilities for EVT6 classes 0 and 1.  This is
  appropriate when the external task is explicitly defined on the shared
  earthquake/explosion label space.
* ``strict_six_class_output``: the original six-class argmax is retained;
  predictions of classes 2--5 count as errors.  This exposes leakage to
  non-shared DiTing labels instead of silently dropping those samples.

For both protocols the script reports trace-level and event-level (mean
probability) metrics.  No external label is used to choose a threshold,
weight, or aggregation rule.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


CLASS_NAMES = ("eq", "ep")
N_EVT6 = 6


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def binary_metrics(target: np.ndarray, predicted: np.ndarray) -> dict:
    """Metrics for target in {0,1}; predictions outside {0,1} are errors."""
    if target.ndim != 1 or predicted.ndim != 1 or len(target) != len(predicted):
        raise ValueError("target and predicted must be aligned one-dimensional arrays")
    if not np.isin(target, (0, 1)).all():
        raise ValueError("external targets must be restricted to EVT6 classes 0 and 1")

    rows = []
    f1s, precisions, recalls = [], [], []
    for cls, name in enumerate(CLASS_NAMES):
        tp = int(np.sum((target == cls) & (predicted == cls)))
        fp = int(np.sum((target != cls) & (predicted == cls)))
        fn = int(np.sum((target == cls) & (predicted != cls)))
        support = int(np.sum(target == cls))
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        rows.append({
            "class_id": cls,
            "class": name,
            "support": support,
            "predicted": int(np.sum(predicted == cls)),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "num_samples": int(len(target)),
        "accuracy": float(np.mean(target == predicted)) if len(target) else 0.0,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "predicted_nonshared": int(np.sum(~np.isin(predicted, (0, 1)))),
        "per_class": rows,
    }


def write_per_class(path: Path, metrics: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_id", "class", "support", "predicted", "precision", "recall", "f1"],
        )
        writer.writeheader()
        writer.writerows(metrics["per_class"])


def read_predictions(path: Path, event_id_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    required_probs = [f"prob_evt6_{i}" for i in range(N_EVT6)]
    target, probabilities, event_ids = [], [], []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = {"tgt_evt6", *required_probs} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing required columns: {sorted(missing)}")
        if event_id_col not in (reader.fieldnames or []):
            raise ValueError(f"{path}: missing event id column {event_id_col!r}")
        for row_no, row in enumerate(reader, start=2):
            try:
                y = int(float(row["tgt_evt6"]))
                p = [float(row[c]) for c in required_probs]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{row_no}: invalid target or probability") from exc
            if y not in (0, 1):
                raise ValueError(f"{path}:{row_no}: target {y} is not eq/ep")
            if not np.isfinite(p).all():
                raise ValueError(f"{path}:{row_no}: non-finite probability")
            event_id = (row.get(event_id_col) or "").strip()
            if not event_id:
                raise ValueError(f"{path}:{row_no}: empty event id in {event_id_col!r}")
            target.append(y)
            probabilities.append(p)
            event_ids.append(event_id)
    if not target:
        raise ValueError(f"{path}: no rows")
    return np.asarray(target, dtype=np.int64), np.asarray(probabilities, dtype=np.float64), np.asarray(event_ids), event_ids


def event_average(target: np.ndarray, probabilities: np.ndarray, event_ids: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[str, dict] = defaultdict(lambda: {"target": [], "probabilities": []})
    for y, p, event_id in zip(target, probabilities, event_ids):
        grouped[event_id]["target"].append(int(y))
        grouped[event_id]["probabilities"].append(p)

    y_event, p_event = [], []
    for event_id in sorted(grouped):
        item = grouped[event_id]
        labels = set(item["target"])
        if len(labels) != 1:
            raise ValueError(f"event {event_id!r} has inconsistent external labels: {sorted(labels)}")
        y_event.append(labels.pop())
        p_event.append(np.mean(np.asarray(item["probabilities"]), axis=0))
    return np.asarray(y_event, dtype=np.int64), np.asarray(p_event, dtype=np.float64)


def evaluate(target: np.ndarray, probabilities: np.ndarray) -> dict:
    # No re-fitting: comparison within (eq, ep) is equivalent to renormalising
    # their two probabilities and then taking argmax.
    projected = np.argmax(probabilities[:, :2], axis=1).astype(np.int64)
    strict = np.argmax(probabilities, axis=1).astype(np.int64)
    return {
        "shared_label_projection": binary_metrics(target, projected),
        "strict_six_class_output": binary_metrics(target, strict),
        "strict_output_class_counts": {
            str(cls): int(np.sum(strict == cls)) for cls in range(N_EVT6)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_csv", type=Path, help="external inference CSV saved with --save-test-probs true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--event-id-col", default="comcat_event_id")
    args = parser.parse_args()

    target, probabilities, _, event_ids = read_predictions(args.results_csv, args.event_id_col)
    event_target, event_probabilities = event_average(target, probabilities, event_ids)
    report = {
        "protocol": {
            "external_labels": {"0": "earthquake / eq", "1": "explosion / ep"},
            "selection": "No target-domain fine-tuning, threshold tuning, or aggregation-weight selection.",
            "shared_label_projection": "argmax over only prob_evt6_0 and prob_evt6_1",
            "strict_six_class_output": "argmax over all six EVT6 probabilities; classes 2-5 count as errors",
            "event_aggregation": "unweighted arithmetic mean of station-level six-class probabilities",
        },
        "source_csv": str(args.results_csv.resolve()),
        "event_id_column": args.event_id_col,
        "trace": evaluate(target, probabilities),
        "event_mean_probability": evaluate(event_target, event_probabilities),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for level, metrics in (("trace", report["trace"]), ("event_mean_probability", report["event_mean_probability"])):
        for protocol in ("shared_label_projection", "strict_six_class_output"):
            write_per_class(args.output_dir / f"{level}_{protocol}_per_class.csv", metrics[protocol])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

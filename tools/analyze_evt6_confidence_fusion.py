#!/usr/bin/env python3
"""Select confidence-weighted event fusion on validation, then evaluate test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CLASS_NAMES = ("eq", "ep", "ss", "sp", "se", "ot")
PROB_COLS = [f"prob_evt6_{index}" for index in range(len(CLASS_NAMES))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-csv", required=True, type=Path)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--alphas", default="0,0.5,1,2")
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path.expanduser().resolve())
    required = ["event_uid", "tgt_evt6", *PROB_COLS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    inconsistent = frame.groupby("event_uid")["tgt_evt6"].nunique()
    if (inconsistent != 1).any():
        raise ValueError(f"{path}: an event has multiple target labels")
    return frame


def report(target: np.ndarray, pred: np.ndarray):
    num_classes = len(CLASS_NAMES)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_id, pred_id in zip(target, pred):
        cm[int(true_id), int(pred_id)] += 1
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros(num_classes), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(num_classes), where=support != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(num_classes),
        where=(precision + recall) != 0,
    )
    summary = {
        "num_events": int(cm.sum()),
        "accuracy": float(tp.sum() / cm.sum()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }
    per_class = pd.DataFrame(
        {
            "class_id": np.arange(num_classes),
            "class": CLASS_NAMES,
            "support": support,
            "predicted": predicted,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )
    return summary, per_class, cm


def fuse_events(frame: pd.DataFrame, alpha: float):
    event_ids = []
    targets = []
    predictions = []
    fused_probabilities = []
    for event_uid, group in frame.groupby("event_uid", sort=True):
        probabilities = group[PROB_COLS].to_numpy(dtype=np.float64)
        confidence = probabilities.max(axis=1)
        weights = np.power(np.clip(confidence, 1e-12, 1.0), float(alpha))
        fused = np.average(probabilities, axis=0, weights=weights)
        event_ids.append(event_uid)
        targets.append(int(group["tgt_evt6"].iloc[0]))
        predictions.append(int(np.argmax(fused)))
        fused_probabilities.append(fused)
    return (
        np.asarray(event_ids),
        np.asarray(targets, dtype=int),
        np.asarray(predictions, dtype=int),
        np.asarray(fused_probabilities, dtype=np.float64),
    )


def main() -> None:
    args = parse_args()
    alphas = sorted({float(value.strip()) for value in args.alphas.split(",") if value.strip()})
    if not alphas or any(value < 0 for value in alphas):
        raise ValueError("--alphas must contain non-negative values")

    val_frame = load_predictions(args.val_csv)
    test_frame = load_predictions(args.test_csv)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for alpha in alphas:
        _, target, pred, _ = fuse_events(val_frame, alpha)
        summary, _, _ = report(target, pred)
        candidates.append({"alpha": alpha, **summary})
    candidates_frame = pd.DataFrame(candidates)
    candidates_frame.to_csv(out_dir / "validation_alpha_selection.csv", index=False)

    # Select only on validation: macro-F1, then accuracy, then the smaller alpha.
    selected = max(
        candidates,
        key=lambda row: (row["macro_f1"], row["accuracy"], -row["alpha"]),
    )
    selected_alpha = float(selected["alpha"])

    event_ids, target, pred, fused_probs = fuse_events(test_frame, selected_alpha)
    test_summary, per_class, cm = report(target, pred)
    per_class.to_csv(out_dir / "test_per_class_metrics.csv", index=False)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        out_dir / "test_confusion_matrix.csv", index_label="true_class"
    )

    event_predictions = pd.DataFrame(
        {
            "event_uid": event_ids,
            "target": target,
            "prediction": pred,
        }
    )
    for index, class_name in enumerate(CLASS_NAMES):
        event_predictions[f"prob_{class_name}"] = fused_probs[:, index]
    event_predictions.to_csv(out_dir / "test_event_predictions.csv", index=False)

    output = {
        "method": "max_probability_confidence_weighted_mean",
        "weight_formula": "weight_i = max(prob_i) ** alpha",
        "class_order": list(CLASS_NAMES),
        "validation_candidates": candidates,
        "selected_alpha_from_validation": selected_alpha,
        "selected_validation_metrics": selected,
        "test_metrics": test_summary,
        "val_csv": str(args.val_csv.expanduser().resolve()),
        "test_csv": str(args.test_csv.expanduser().resolve()),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Analysis files saved to: {out_dir}")


if __name__ == "__main__":
    main()

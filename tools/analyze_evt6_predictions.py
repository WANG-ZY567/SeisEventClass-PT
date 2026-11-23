#!/usr/bin/env python3
"""Create reproducible trace- and event-level EVT6 metrics from test predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CLASS_NAMES = ("eq", "ep", "ss", "sp", "se", "ot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="test_results_diting2_evt6_test.csv")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output directory (default: <run_dir>/analysis)",
    )
    return parser.parse_args()


def classification_report(target: np.ndarray, pred: np.ndarray):
    num_classes = len(CLASS_NAMES)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_id, pred_id in zip(target, pred):
        cm[int(true_id), int(pred_id)] += 1

    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    precision = np.divide(
        tp, predicted, out=np.zeros(num_classes, dtype=float), where=predicted != 0
    )
    recall = np.divide(
        tp, support, out=np.zeros(num_classes, dtype=float), where=support != 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(num_classes, dtype=float),
        where=(precision + recall) != 0,
    )
    accuracy = float(tp.sum() / cm.sum()) if cm.sum() else 0.0

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
    summary = {
        "num_samples": int(cm.sum()),
        "accuracy": accuracy,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }
    return summary, per_class, cm


def save_report(out_dir: Path, prefix: str, target: np.ndarray, pred: np.ndarray):
    summary, per_class, cm = classification_report(target, pred)
    per_class.to_csv(out_dir / f"{prefix}_per_class_metrics.csv", index=False)

    cm_frame = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_frame.index.name = "true_class"
    cm_frame.to_csv(out_dir / f"{prefix}_confusion_matrix.csv")

    support = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm,
        support,
        out=np.zeros_like(cm, dtype=float),
        where=support != 0,
    )
    cm_norm_frame = pd.DataFrame(cm_norm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_norm_frame.index.name = "true_class"
    cm_norm_frame.to_csv(out_dir / f"{prefix}_confusion_matrix_normalized.csv")
    return summary, per_class, cm_frame


def main() -> None:
    args = parse_args()
    csv_path = args.csv.expanduser().resolve()
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir is not None
        else csv_path.parent / "analysis"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(csv_path)
    prob_cols = [f"prob_evt6_{index}" for index in range(len(CLASS_NAMES))]
    required = ["event_uid", "tgt_evt6", "pred_evt6", *prob_cols]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    trace_target = frame["tgt_evt6"].astype(int).to_numpy()
    trace_pred = frame["pred_evt6"].astype(int).to_numpy()
    trace_summary, _, _ = save_report(
        out_dir, "trace", trace_target, trace_pred
    )

    inconsistent = frame.groupby("event_uid")["tgt_evt6"].nunique()
    if (inconsistent != 1).any():
        bad = inconsistent[inconsistent != 1].index.tolist()[:10]
        raise ValueError(f"events with inconsistent target labels: {bad}")

    grouped = frame.groupby("event_uid", sort=True)
    event_target = grouped["tgt_evt6"].first().astype(int)
    event_mean_prob = grouped[prob_cols].mean()
    event_pred_mean = np.argmax(event_mean_prob.to_numpy(), axis=1)

    event_pred_vote = []
    for _, group in grouped:
        counts = np.bincount(group["pred_evt6"].astype(int), minlength=len(CLASS_NAMES))
        candidates = np.flatnonzero(counts == counts.max())
        if len(candidates) == 1:
            event_pred_vote.append(int(candidates[0]))
        else:
            means = group[prob_cols].mean().to_numpy()
            event_pred_vote.append(int(candidates[np.argmax(means[candidates])]))
    event_pred_vote = np.asarray(event_pred_vote, dtype=int)

    event_predictions = event_mean_prob.copy()
    event_predictions.insert(0, "target", event_target.to_numpy())
    event_predictions["pred_mean_probability"] = event_pred_mean
    event_predictions["pred_majority_vote"] = event_pred_vote
    event_predictions.to_csv(out_dir / "event_predictions.csv")

    mean_summary, _, _ = save_report(
        out_dir,
        "event_mean_probability",
        event_target.to_numpy(),
        event_pred_mean,
    )
    vote_summary, _, _ = save_report(
        out_dir,
        "event_majority_vote",
        event_target.to_numpy(),
        event_pred_vote,
    )

    summary = {
        "source_csv": str(csv_path),
        "class_order": list(CLASS_NAMES),
        "trace": trace_summary,
        "event_mean_probability": mean_summary,
        "event_majority_vote": vote_summary,
        "event_strategy_predictions_identical": bool(
            np.array_equal(event_pred_mean, event_pred_vote)
        ),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Analysis files saved to: {out_dir}")


if __name__ == "__main__":
    main()

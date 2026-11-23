#!/usr/bin/env python3
"""Select a two-checkpoint probability mixture on validation, then score test once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_evt6_predictions import CLASS_NAMES, classification_report, save_report


PROB_COLS = [f"prob_evt6_{i}" for i in range(len(CLASS_NAMES))]
REQUIRED = ["trace_uid", "event_uid", "tgt_evt6", *PROB_COLS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-old", required=True, type=Path)
    parser.add_argument("--val-new", required=True, type=Path)
    parser.add_argument("--test-old", required=True, type=Path)
    parser.add_argument("--test-new", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--new-weights",
        default="0,0.25,0.5,0.75,1",
        help="Validation grid for new-checkpoint weight (default: 0,0.25,0.5,0.75,1)",
    )
    return parser.parse_args()


def load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path.expanduser().resolve(), dtype={"trace_uid": "string", "event_uid": "string"})
    missing = [name for name in REQUIRED if name not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")
    if frame.empty or frame[["trace_uid", "event_uid"]].isna().any().any():
        raise ValueError(f"{path}: empty file or missing trace/event IDs")
    if frame["trace_uid"].duplicated().any():
        raise ValueError(f"{path}: duplicate trace_uid values")

    labels = pd.to_numeric(frame["tgt_evt6"], errors="raise").to_numpy()
    if not np.all(np.isfinite(labels)) or not np.all(labels == labels.astype(int)):
        raise ValueError(f"{path}: invalid target labels")
    if not np.all((labels >= 0) & (labels < len(CLASS_NAMES))):
        raise ValueError(f"{path}: target labels outside 0..{len(CLASS_NAMES) - 1}")
    frame["tgt_evt6"] = labels.astype(int)

    probabilities = frame[PROB_COLS].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{path}: non-finite probabilities")
    if (probabilities < -1e-6).any() or (probabilities > 1 + 1e-6).any():
        raise ValueError(f"{path}: probabilities outside [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-3, rtol=0):
        raise ValueError(f"{path}: probability rows do not sum to one")
    return frame


def load_aligned_pair(old_path: Path, new_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    old = load_frame(old_path)
    new = load_frame(new_path)
    old_ids = set(old["trace_uid"])
    new_ids = set(new["trace_uid"])
    if old_ids != new_ids:
        raise ValueError(
            f"Trace sets differ: old-only={len(old_ids - new_ids)}, "
            f"new-only={len(new_ids - old_ids)}"
        )
    new = new.set_index("trace_uid").loc[old["trace_uid"].tolist()].reset_index()
    for column in ("event_uid", "tgt_evt6"):
        if not old[column].reset_index(drop=True).equals(new[column].reset_index(drop=True)):
            raise ValueError(f"Aligned traces disagree on {column}")
    inconsistent = old.groupby("event_uid")["tgt_evt6"].nunique()
    if (inconsistent != 1).any():
        raise ValueError("An event has more than one target label")
    return (
        old,
        old[PROB_COLS].to_numpy(dtype=np.float64),
        new[PROB_COLS].to_numpy(dtype=np.float64),
    )


def mixture(old_probs: np.ndarray, new_probs: np.ndarray, new_weight: float) -> np.ndarray:
    return (1.0 - new_weight) * old_probs + new_weight * new_probs


def write_predictions(
    path: Path, frame: pd.DataFrame, old_probs: np.ndarray, new_probs: np.ndarray, new_weight: float
) -> tuple[dict, np.ndarray]:
    probabilities = mixture(old_probs, new_probs, new_weight)
    predictions = np.argmax(probabilities, axis=1).astype(int)
    output = frame.copy()
    output["pred_evt6"] = predictions
    for index, name in enumerate(PROB_COLS):
        output[name] = probabilities[:, index]
    output.to_csv(path, index=False)
    summary, _, _ = classification_report(frame["tgt_evt6"].to_numpy(), predictions)
    return summary, predictions


def main() -> None:
    args = parse_args()
    weights = sorted({float(item.strip()) for item in args.new_weights.split(",") if item.strip()})
    if not weights or any(not np.isfinite(weight) or weight < 0 or weight > 1 for weight in weights):
        raise ValueError("--new-weights must contain finite values in [0, 1]")
    if 0.0 not in weights or 1.0 not in weights:
        raise ValueError("Include both 0 and 1 as single-checkpoint controls")

    val_frame, val_old, val_new = load_aligned_pair(args.val_old, args.val_new)
    test_frame, test_old, test_new = load_aligned_pair(args.test_old, args.test_new)
    if set(val_frame["trace_uid"]) & set(test_frame["trace_uid"]):
        raise ValueError("Validation and test share trace_uid values")
    if set(val_frame["event_uid"]) & set(test_frame["event_uid"]):
        raise ValueError("Validation and test share event_uid values")

    val_target = val_frame["tgt_evt6"].to_numpy(dtype=int)
    candidates = []
    for weight in weights:
        prediction = np.argmax(mixture(val_old, val_new, weight), axis=1)
        metrics, _, _ = classification_report(val_target, prediction)
        candidates.append({"new_weight": weight, **metrics})

    # The test labels are never used to select a weight. Favor the older model on a full tie.
    selected = max(candidates, key=lambda row: (row["accuracy"], row["macro_f1"], -row["new_weight"]))
    weight = selected["new_weight"]
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(candidates).to_csv(out_dir / "validation_weight_grid.csv", index=False)

    val_metrics, val_pred = write_predictions(
        out_dir / "val_fused_predictions.csv", val_frame, val_old, val_new, weight
    )
    test_metrics, test_pred = write_predictions(
        out_dir / "test_fused_predictions.csv", test_frame, test_old, test_new, weight
    )
    save_report(out_dir, "val_trace", val_target, val_pred)
    save_report(out_dir, "test_trace", test_frame["tgt_evt6"].to_numpy(dtype=int), test_pred)

    summary = {
        "method": "fixed_two_checkpoint_probability_mixture",
        "formula": "fused_probability = (1 - new_weight) * old + new_weight * new",
        "class_order": list(CLASS_NAMES),
        "selection_rule": "maximize validation trace accuracy, then macro-F1, then prefer smaller new weight",
        "validation_candidates": candidates,
        "selected_new_weight": weight,
        "selected_validation_trace": val_metrics,
        "test_trace": test_metrics,
        "num_validation_events": int(val_frame["event_uid"].nunique()),
        "num_test_events": int(test_frame["event_uid"].nunique()),
        "source_csvs": {
            "val_old": str(args.val_old.expanduser().resolve()),
            "val_new": str(args.val_new.expanduser().resolve()),
            "test_old": str(args.test_old.expanduser().resolve()),
            "test_new": str(args.test_new.expanduser().resolve()),
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved fusion results to: {out_dir}")


if __name__ == "__main__":
    main()

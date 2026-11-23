#!/usr/bin/env python3
"""Select small EVT6 class-logit biases on validation predictions and apply to test.

This is a post-hoc calibration step, not an ensemble.  The calibrated probability is

    softmax(log(original_probability) + class_bias).

Only validation labels are used to select the biases.  Test labels are used once, after
selection, to report the final held-out metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_evt6_predictions import CLASS_NAMES, classification_report, save_report


PROBABILITY_COLUMNS = [f"prob_evt6_{index}" for index in range(len(CLASS_NAMES))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--tuned-classes",
        default="ep,ss,sp,ot",
        help="comma-separated classes whose biases may change (default: ep,ss,sp,ot)",
    )
    parser.add_argument("--bias-min", type=float, default=-0.30)
    parser.add_argument("--bias-max", type=float, default=0.30)
    parser.add_argument("--bias-step", type=float, default=0.025)
    parser.add_argument("--max-passes", type=int, default=10)
    return parser.parse_args()


def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ["event_uid", "tgt_evt6", *PROBABILITY_COLUMNS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")

    probabilities = frame[PROBABILITY_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{path}: probabilities contain NaN or infinity")
    if (probabilities < 0).any():
        raise ValueError(f"{path}: probabilities contain negative values")
    if (probabilities.sum(axis=1) <= 0).any():
        raise ValueError(f"{path}: at least one probability row has zero mass")
    return frame


def logits_from_frame(frame: pd.DataFrame) -> np.ndarray:
    probabilities = frame[PROBABILITY_COLUMNS].to_numpy(dtype=np.float64)
    probabilities = np.clip(probabilities, 1e-12, None)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return np.log(probabilities)


def probabilities_from_logits(log_probabilities: np.ndarray, bias: np.ndarray) -> np.ndarray:
    adjusted = log_probabilities + bias[None, :]
    adjusted -= adjusted.max(axis=1, keepdims=True)
    probabilities = np.exp(adjusted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def trace_metrics(
    log_probabilities: np.ndarray, target: np.ndarray, bias: np.ndarray
) -> dict[str, float | int]:
    prediction = np.argmax(log_probabilities + bias[None, :], axis=1)
    summary, _, _ = classification_report(target, prediction)
    return summary


def score(metrics: dict[str, float | int], bias: np.ndarray) -> tuple[float, float, float]:
    return (
        float(metrics["accuracy"]),
        float(metrics["macro_f1"]),
        -float(np.dot(bias, bias)),
    )


def coordinate_search(
    log_probabilities: np.ndarray,
    target: np.ndarray,
    tuned_indices: list[int],
    grid: np.ndarray,
    max_passes: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    bias = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    history: list[dict[str, object]] = []

    initial_metrics = trace_metrics(log_probabilities, target, bias)
    history.append(
        {
            "pass": 0,
            "class": "initial",
            "biases": bias.tolist(),
            "metrics": initial_metrics,
        }
    )

    for pass_index in range(1, max_passes + 1):
        changed = False
        for class_index in tuned_indices:
            best_bias = bias.copy()
            best_metrics = trace_metrics(log_probabilities, target, best_bias)
            best_score = score(best_metrics, best_bias)

            for value in grid:
                candidate = bias.copy()
                candidate[class_index] = float(value)
                metrics = trace_metrics(log_probabilities, target, candidate)
                candidate_score = score(metrics, candidate)
                if candidate_score > best_score:
                    best_bias = candidate
                    best_metrics = metrics
                    best_score = candidate_score

            if not np.isclose(best_bias[class_index], bias[class_index]):
                changed = True
            bias = best_bias
            history.append(
                {
                    "pass": pass_index,
                    "class": CLASS_NAMES[class_index],
                    "biases": bias.tolist(),
                    "metrics": best_metrics,
                }
            )

        if not changed:
            break

    return bias, history


def calibrated_frame(frame: pd.DataFrame, bias: np.ndarray) -> pd.DataFrame:
    output = frame.copy()
    probabilities = probabilities_from_logits(logits_from_frame(frame), bias)
    output.loc[:, PROBABILITY_COLUMNS] = probabilities
    output["pred_evt6"] = np.argmax(probabilities, axis=1)
    return output


def analyze_frame(frame: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_target = frame["tgt_evt6"].astype(int).to_numpy()
    trace_prediction = frame["pred_evt6"].astype(int).to_numpy()
    trace_summary, _, _ = save_report(
        out_dir, "trace", trace_target, trace_prediction
    )

    inconsistent = frame.groupby("event_uid")["tgt_evt6"].nunique()
    if (inconsistent != 1).any():
        bad = inconsistent[inconsistent != 1].index.tolist()[:10]
        raise ValueError(f"events with inconsistent target labels: {bad}")

    grouped = frame.groupby("event_uid", sort=True)
    event_target = grouped["tgt_evt6"].first().astype(int).to_numpy()
    event_mean_probability = grouped[PROBABILITY_COLUMNS].mean().to_numpy()
    event_mean_prediction = np.argmax(event_mean_probability, axis=1)
    mean_summary, _, _ = save_report(
        out_dir,
        "event_mean_probability",
        event_target,
        event_mean_prediction,
    )

    event_vote_prediction = []
    for _, group in grouped:
        counts = np.bincount(
            group["pred_evt6"].astype(int), minlength=len(CLASS_NAMES)
        )
        candidates = np.flatnonzero(counts == counts.max())
        if len(candidates) == 1:
            event_vote_prediction.append(int(candidates[0]))
        else:
            means = group[PROBABILITY_COLUMNS].mean().to_numpy()
            event_vote_prediction.append(
                int(candidates[np.argmax(means[candidates])])
            )
    event_vote_prediction = np.asarray(event_vote_prediction, dtype=int)
    vote_summary, _, _ = save_report(
        out_dir,
        "event_majority_vote",
        event_target,
        event_vote_prediction,
    )
    return {
        "trace": trace_summary,
        "event_mean_probability": mean_summary,
        "event_majority_vote": vote_summary,
        "event_strategy_predictions_identical": bool(
            np.array_equal(event_mean_prediction, event_vote_prediction)
        ),
    }


def main() -> None:
    args = parse_args()
    if args.bias_step <= 0:
        raise ValueError("--bias-step must be positive")
    if args.bias_min > args.bias_max:
        raise ValueError("--bias-min must not exceed --bias-max")
    if args.max_passes <= 0:
        raise ValueError("--max-passes must be positive")

    class_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
    tuned_names = [name.strip() for name in args.tuned_classes.split(",") if name.strip()]
    unknown = [name for name in tuned_names if name not in class_to_index]
    if unknown:
        raise ValueError(f"unknown classes in --tuned-classes: {unknown}")
    if len(set(tuned_names)) != len(tuned_names):
        raise ValueError("--tuned-classes contains duplicates")
    tuned_indices = [class_to_index[name] for name in tuned_names]

    val_path = args.val_csv.expanduser().resolve()
    test_path = args.test_csv.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    validation = load_predictions(val_path)
    test = load_predictions(test_path)
    validation_target = validation["tgt_evt6"].astype(int).to_numpy()
    validation_log_probabilities = logits_from_frame(validation)

    count = int(np.floor((args.bias_max - args.bias_min) / args.bias_step + 0.5))
    grid = args.bias_min + np.arange(count + 1, dtype=np.float64) * args.bias_step
    grid = grid[grid <= args.bias_max + 1e-12]
    if not np.any(np.isclose(grid, 0.0)):
        grid = np.sort(np.append(grid, 0.0))

    selected_bias, history = coordinate_search(
        validation_log_probabilities,
        validation_target,
        tuned_indices,
        grid,
        args.max_passes,
    )

    calibrated_validation = calibrated_frame(validation, selected_bias)
    calibrated_test = calibrated_frame(test, selected_bias)
    val_output = out_dir / "val_calibrated_predictions.csv"
    test_output = out_dir / "test_calibrated_predictions.csv"
    calibrated_validation.to_csv(val_output, index=False)
    calibrated_test.to_csv(test_output, index=False)

    uncalibrated_val = calibrated_frame(validation, np.zeros(len(CLASS_NAMES)))
    uncalibrated_test = calibrated_frame(test, np.zeros(len(CLASS_NAMES)))
    baseline_val_metrics = analyze_frame(uncalibrated_val, out_dir / "val_before")
    calibrated_val_metrics = analyze_frame(calibrated_validation, out_dir / "val_after")
    baseline_test_metrics = analyze_frame(uncalibrated_test, out_dir / "test_before")
    calibrated_test_metrics = analyze_frame(calibrated_test, out_dir / "test_after")

    summary = {
        "method": "single_model_additive_class_logit_bias",
        "formula": "softmax(log(original_probability) + class_bias)",
        "selection_data": "validation only",
        "selection_rule": (
            "maximize validation trace accuracy, then validation macro-F1, "
            "then prefer smaller bias L2 norm"
        ),
        "class_order": list(CLASS_NAMES),
        "tuned_classes": tuned_names,
        "fixed_classes": [name for name in CLASS_NAMES if name not in tuned_names],
        "search_grid": {
            "minimum": args.bias_min,
            "maximum": args.bias_max,
            "step": args.bias_step,
            "max_coordinate_passes": args.max_passes,
        },
        "selected_biases": {
            name: float(selected_bias[index])
            for index, name in enumerate(CLASS_NAMES)
        },
        "validation_before": baseline_val_metrics,
        "validation_after": calibrated_val_metrics,
        "test_before": baseline_test_metrics,
        "test_after": calibrated_test_metrics,
        "source_csvs": {"validation": str(val_path), "test": str(test_path)},
        "output_csvs": {"validation": str(val_output), "test": str(test_output)},
        "coordinate_search_history": history,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Calibration files saved to: {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate a fixed EVT6 checkpoint on original and time-masked validation waves.

The script never trains, evaluates the test split, or edits the source dataset.
Both masks remove 400 samples before the normal inference normalization.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROB_COLS = [f"prob_evt6_{i}" for i in range(6)]
MASKS = {"pre_p": (0, 400), "late": (3500, 3900)}


def make_masked_data(source: Path, manifest: pd.DataFrame, target: Path,
                     interval: tuple[int, int]) -> None:
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit data: {target}")
    waves = target / "waves"
    waves.mkdir(parents=True)
    paths = []
    for index, relative_path in enumerate(manifest["_npy_path"].astype(str)):
        path = source / relative_path
        wave = np.load(path)
        if wave.shape != (3, 8192):
            raise ValueError(f"{path}: expected (3,8192), got {wave.shape}")
        wave = wave.astype(np.float32, copy=True)
        wave[:, interval[0]:interval[1]] = 0
        name = f"{index:06d}.npy"
        np.save(waves / name, wave)
        paths.append(f"waves/{name}")
    masked = manifest.copy()
    masked["_npy_path"] = paths
    masked.to_csv(target / "meta_evt6_val.csv", index=False)


def evaluate(repo: Path, data: Path, checkpoint: Path, out: Path,
             workers: int, device: str) -> Path:
    run_base = out / "runs"
    run_base.mkdir(parents=True)
    command = [
        sys.executable, str(repo / "main.py"), "--mode", "test",
        "--eval-split", "val", "--model-name", "SeisMoLLM_evt6",
        "--data", str(data), "--checkpoint", str(checkpoint),
        "--in-samples", "8192", "--seed", "20261105",
        "--batch-size", "16", "--norm-mode", "std",
        "--workers", str(workers), "--device", device,
        "--save-test-probs", "true", "--force-logdir", "true",
        "--loss-name", "class_balanced_ce", "--cb-beta", "0.995",
        "--cb-samples-per-class", "4172,4450,4233,1095,148,2381",
        "--log-base", str(run_base),
    ]
    with (out / "launcher.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)
    results = list(run_base.glob("*/val_results_diting2_evt6_val.csv"))
    if len(results) != 1:
        raise RuntimeError(f"Expected one validation CSV in {run_base}, got {results}")
    return results[0]


def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    needed = {"trace_uid", "event_uid", "tgt_evt6", "pred_evt6", *PROB_COLS}
    missing = sorted(needed - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing {missing}")
    return frame.set_index("trace_uid", verify_integrity=True).sort_index()


def scores(frame: pd.DataFrame) -> dict:
    trace_correct = frame["tgt_evt6"].to_numpy() == frame["pred_evt6"].to_numpy()
    grouped = frame.groupby("event_uid", sort=True)
    if (grouped["tgt_evt6"].nunique() != 1).any():
        raise ValueError("Mixed target labels within an event")
    event_target = grouped["tgt_evt6"].first().to_numpy(dtype=int)
    event_pred = grouped[PROB_COLS].mean().to_numpy().argmax(axis=1)
    return {
        "trace_accuracy": float(trace_correct.mean()),
        "event_mean_probability_accuracy": float((event_target == event_pred).mean()),
        "num_traces": len(frame),
        "num_events": len(event_target),
    }


def compare(base: pd.DataFrame, masked: pd.DataFrame) -> dict:
    if not base.index.equals(masked.index):
        raise ValueError("Baseline and masked trace IDs differ")
    if not base["tgt_evt6"].equals(masked["tgt_evt6"]):
        raise ValueError("Baseline and masked labels differ")
    before = base["pred_evt6"].to_numpy() == base["tgt_evt6"].to_numpy()
    after = masked["pred_evt6"].to_numpy() == masked["tgt_evt6"].to_numpy()
    old = scores(base)
    new = scores(masked)
    return {
        "masked_scores": new,
        "trace_accuracy_change_pp": 100 * (new["trace_accuracy"] - old["trace_accuracy"]),
        "event_accuracy_change_pp": 100 * (
            new["event_mean_probability_accuracy"] - old["event_mean_probability_accuracy"]),
        "prediction_changed": int((base["pred_evt6"] != masked["pred_evt6"]).sum()),
        "correct_to_wrong": int((before & ~after).sum()),
        "wrong_to_correct": int((~before & after).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    source = args.data_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    out = args.out_dir.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if (out / "summary.json").exists():
        raise FileExistsError(f"Audit already completed: {out}")
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(source / "meta_evt6_val.csv", low_memory=False)
    if manifest["trace_uid"].duplicated().any():
        raise ValueError("Duplicate validation trace_uid")
    print("Evaluating original validation waves", flush=True)
    base = load_predictions(evaluate(repo, source, checkpoint, out / "original",
                                     args.workers, args.device))
    report = {"checkpoint": str(checkpoint), "baseline": scores(base), "masks": {}}
    for name, interval in MASKS.items():
        print(f"Preparing and evaluating {name}: {interval}", flush=True)
        variant_data = out / name / "data"
        make_masked_data(source, manifest, variant_data, interval)
        masked = load_predictions(evaluate(repo, variant_data, checkpoint,
                                           out / name, args.workers, args.device))
        report["masks"][name] = {"interval_samples": list(interval),
                                  **compare(base, masked)}
    output = json.dumps(report, ensure_ascii=False, indent=2)
    (out / "summary.json").write_text(output + "\n", encoding="utf-8")
    print(output, flush=True)


if __name__ == "__main__":
    main()

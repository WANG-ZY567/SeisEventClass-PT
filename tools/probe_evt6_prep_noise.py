#!/usr/bin/env python3
"""Diagnose whether pre-P noise predicts the raw HDF5 source on event-disjoint splits.

This is a small diagnostic classifier. It does not modify EVT6 data or checkpoints.
The interval [0, 400) is P-10 s through P-2 s at 50 Hz, leaving a two-second
buffer before Pg to reduce possible zero-phase filter leakage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def sample_manifest(path: Path, per_source: int, seed: int) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = {"_src", "_npy_path", "event_uid", "p_pick"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    if set(frame["_src"].dropna()) != {"natural", "non"}:
        raise ValueError(f"{path}: expected natural/non source values")
    if not (pd.to_numeric(frame["p_pick"], errors="coerce") >= 500).all():
        raise ValueError(f"{path}: expected Pg at or after sample 500")
    return pd.concat(
        [group.sample(n=min(per_source, len(group)), random_state=seed)
         for _, group in frame.groupby("_src", sort=True)],
        ignore_index=True,
    )


def features(wave_path: Path) -> tuple[np.ndarray, np.ndarray]:
    wave = np.load(wave_path, mmap_mode="r")
    if wave.shape != (3, 8192):
        raise ValueError(f"{wave_path}: expected (3,8192), got {wave.shape}")
    wave = np.asarray(wave, dtype=np.float64)
    segment = wave[:, :400]
    centered = segment - segment.mean(axis=1, keepdims=True)
    local_std = centered.std(axis=1, keepdims=True)
    if not np.isfinite(centered).all() or np.any(local_std <= 0):
        raise ValueError(f"{wave_path}: invalid pre-P samples")
    normalized = centered / local_std
    full_std = wave.std(axis=1)
    freq = np.fft.rfftfreq(segment.shape[1], d=1 / 50)
    power = np.abs(np.fft.rfft(normalized, axis=1)) ** 2
    power = power[:, freq >= 1]
    freq = freq[freq >= 1]
    power /= power.sum(axis=1, keepdims=True).clip(min=1e-20)
    shape = []
    relative = []
    for channel in range(3):
        z = normalized[channel]
        p = power[channel]
        shape.extend([
            *(float(p[(freq >= low) & (freq < high)].sum())
              for low, high in ((1, 5), (5, 10), (10, 20))),
            float((p * freq).sum()),
            float(np.mean(z ** 3)),
            float(np.mean(z ** 4)),
            float(np.mean(z[1:] * z[:-1])),
            float(np.mean(z[5:] * z[:-5])),
            float(np.mean(np.signbit(z[1:]) != np.signbit(z[:-1]))),
        ])
        relative.append(float(local_std[channel, 0] / max(full_std[channel], 1e-20)))
    return np.asarray(shape), np.asarray(relative)


def extract(data_dir: Path, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape_features = []
    relative_features = []
    for relative_path in frame["_npy_path"].astype(str):
        shape, relative = features(data_dir / relative_path)
        shape_features.append(shape)
        relative_features.append(relative)
    return (np.stack(shape_features), np.stack(relative_features),
            (frame["_src"] == "natural").to_numpy(dtype=int))


def fit_and_score(train_x: np.ndarray, train_y: np.ndarray,
                  val_x: np.ndarray, val_y: np.ndarray) -> dict:
    model = make_pipeline(StandardScaler(), LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=0))
    model.fit(train_x, train_y)
    probability = model.predict_proba(val_x)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    return {
        "validation_auc": float(roc_auc_score(val_y, probability)),
        "validation_balanced_accuracy": float(balanced_accuracy_score(val_y, prediction)),
        "validation_accuracy": float(accuracy_score(val_y, prediction)),
        "validation_natural_recall": float(prediction[val_y == 1].mean()),
        "validation_non_recall": float(1 - prediction[val_y == 0].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--train-per-source", type=int, default=2500)
    parser.add_argument("--val-per-source", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()
    train = sample_manifest(args.data_dir / "meta_evt6_train.csv",
                            args.train_per_source, args.seed)
    val = sample_manifest(args.data_dir / "meta_evt6_val.csv",
                          args.val_per_source, args.seed)
    overlap = set(train["event_uid"]) & set(val["event_uid"])
    if overlap:
        raise ValueError(f"train/val event overlap: {len(overlap)}")
    train_shape, train_relative, train_y = extract(args.data_dir, train)
    val_shape, val_relative, val_y = extract(args.data_dir, val)
    report = {
        "interval": "P-10s to P-2s, excluding the last 2s before Pg",
        "train_rows": len(train),
        "val_rows": len(val),
        "train_events": int(train["event_uid"].nunique()),
        "val_events": int(val["event_uid"].nunique()),
        "local_shape_only": fit_and_score(train_shape, train_y, val_shape, val_y),
        "shape_plus_noise_to_full_wave_ratio": fit_and_score(
            np.column_stack((train_shape, train_relative)), train_y,
            np.column_stack((val_shape, val_relative)), val_y),
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create a tiny deterministic EVT6 dataset for pipeline smoke tests only."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--samples", type=int, default=1024, help="waveform length (default: 1024)")
    parser.add_argument("--train-per-class", type=int, default=2)
    parser.add_argument("--val-per-class", type=int, default=1)
    parser.add_argument("--test-per-class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _waveform(class_id: int, index: int, samples: int, rng: np.random.Generator) -> np.ndarray:
    time = np.linspace(0.0, 1.0, samples, endpoint=False, dtype=np.float32)
    frequency = float(class_id + 1)
    channels = []
    for channel in range(3):
        phase = 0.2 * channel + 0.05 * index
        signal = np.sin(2.0 * np.pi * frequency * time + phase)
        signal += 0.05 * rng.normal(size=samples)
        channels.append(signal.astype(np.float32))
    return np.stack(channels, axis=0)


def main() -> None:
    args = _args()
    if args.samples < 1024:
        raise ValueError("--samples must be at least 1024 for the default convolutional head")

    out_dir = args.out_dir.expanduser().resolve()
    waves_dir = out_dir / "waves"
    waves_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    all_rows = []

    counts = {
        "train": args.train_per_class,
        "val": args.val_per_class,
        "test": args.test_per_class,
    }
    for split, per_class in counts.items():
        rows = []
        for class_id in range(6):
            for index in range(per_class):
                event_uid = f"synthetic_{split}_c{class_id}_e{index}"
                key = event_uid
                relative = Path("waves") / f"{key}_0.npy"
                np.save(out_dir / relative, _waveform(class_id, index, args.samples, rng))
                row = {
                    "key": key,
                    "part": 0,
                    "_src": "synthetic",
                    "_rid": key,
                    "_evtype": ("eq", "ep", "ss", "sp", "se", "ot")[class_id],
                    "_evt6": class_id,
                    "_npy_path": relative.as_posix(),
                    "event_uid": event_uid,
                }
                rows.append(row)
                all_rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / f"meta_evt6_{split}.csv", index=False)

    pd.DataFrame(all_rows).to_csv(out_dir / "meta_evt6.csv", index=False)
    print(f"Synthetic EVT6 data written to: {out_dir}")
    print(f"Rows: {len(all_rows)}, waveform shape: [3, {args.samples}]")
    print("This dataset is for software testing only, not scientific evaluation.")


if __name__ == "__main__":
    main()

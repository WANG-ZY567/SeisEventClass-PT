#!/usr/bin/env python3
"""Validate EVT6 split membership, class counts and event leakage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


SPLITS = ("train", "val", "test")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="directory containing meta_evt6_{train,val,test}.csv")
    parser.add_argument(
        "--event-column",
        default="event_uid",
        help="event identifier used to detect leakage (default: event_uid)",
    )
    parser.add_argument("--check-files", action="store_true", help="also verify every _npy_path")
    return parser.parse_args()


def _wave_path(data_dir: Path, raw: object) -> Path:
    path = Path(str(raw).strip())
    return path if path.is_absolute() else data_dir / path


def main() -> int:
    args = _parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    frames: dict[str, pd.DataFrame] = {}
    errors: list[str] = []

    for split in SPLITS:
        path = data_dir / f"meta_evt6_{split}.csv"
        if not path.exists():
            errors.append(f"missing split file: {path}")
            continue
        frame = pd.read_csv(path, low_memory=False)
        required = {"key", "part", "_evt6", "_npy_path"}
        missing = required.difference(frame.columns)
        if missing:
            errors.append(f"{path.name} missing columns: {sorted(missing)}")
            continue
        frames[split] = frame
        counts = frame["_evt6"].value_counts().sort_index().to_dict()
        print(f"{split:>5}: rows={len(frame):>8}, class_counts={counts}")

        duplicate_rows = int(frame.duplicated(subset=["key", "part"]).sum())
        if duplicate_rows:
            errors.append(f"{split}: {duplicate_rows} duplicate key/part rows")

        if args.check_files:
            missing_files = sum(
                not _wave_path(data_dir, value).exists() for value in frame["_npy_path"]
            )
            if missing_files:
                errors.append(f"{split}: {missing_files} waveform files are missing")

    if len(frames) == len(SPLITS):
        row_sets = {
            split: set(zip(frame["key"].astype(str), frame["part"].astype(str)))
            for split, frame in frames.items()
        }
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = row_sets[left].intersection(row_sets[right])
            if overlap:
                errors.append(f"record overlap {left}/{right}: {len(overlap)}")

        if all(args.event_column in frame.columns for frame in frames.values()):
            event_sets = {
                split: set(frame[args.event_column].dropna().astype(str))
                for split, frame in frames.items()
            }
            for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
                overlap = event_sets[left].intersection(event_sets[right])
                if overlap:
                    errors.append(f"event overlap {left}/{right}: {len(overlap)}")
        else:
            errors.append(
                f"cannot audit event leakage: column {args.event_column!r} is absent; "
                "regenerate metadata with --add_xapp_meta"
            )

    if errors:
        print("\n[FAIL]")
        for error in errors:
            print(f"- {error}")
        return 1

    print("\n[OK] Split files are disjoint by record and event.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

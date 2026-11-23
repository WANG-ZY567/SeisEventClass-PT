"""Create a fixed-split EVT6 subset with valid P and S arrivals.

The tool writes only filtered metadata. Waveform paths are made absolute, so the
new protocol directory reuses the existing waves rather than duplicating data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CLASS_ORDER = ("eq", "ep", "ss", "sp", "se", "ot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", type=Path, required=True)
    parser.add_argument("--dst-dir", type=Path, required=True)
    parser.add_argument("--wave-length", type=int, default=8192)
    parser.add_argument(
        "--post-s-samples",
        type=int,
        default=2000,
        help="Require this many samples after S; use 2000 (40 s at 50 Hz).",
    )
    return parser.parse_args()


def as_absolute_path(value: object, src_dir: Path) -> str:
    path = Path(str(value).strip().replace("\\", "/"))
    return str(path if path.is_absolute() else (src_dir / path).resolve())


def counts_by_class(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["_evtype"].value_counts().to_dict()
    return {name: int(counts.get(name, 0)) for name in CLASS_ORDER}


def event_counts_by_class(frame: pd.DataFrame) -> dict[str, int]:
    values = frame.groupby("_evtype")["event_uid"].nunique().to_dict()
    return {name: int(values.get(name, 0)) for name in CLASS_ORDER}


def main() -> None:
    args = parse_args()
    src_dir = args.src_dir.resolve()
    dst_dir = args.dst_dir.resolve()
    if dst_dir.exists() and any(dst_dir.iterdir()):
        raise FileExistsError(
            f"Destination exists and is not empty: {dst_dir}. "
            "Choose a new directory; this tool never overwrites a protocol."
        )
    dst_dir.mkdir(parents=True, exist_ok=False)

    report: dict[str, object] = {
        "source_data_dir": str(src_dir),
        "subset_data_dir": str(dst_dir),
        "criterion": {
            "p_pick": "finite and >= 0",
            "s_pick": "finite and greater than p_pick",
            "wave_length": int(args.wave_length),
            "post_s_samples": int(args.post_s_samples),
        },
        "splits": {},
    }
    event_sets: dict[str, set[str]] = {}

    for split in ("train", "val", "test"):
        input_path = src_dir / f"meta_evt6_{split}.csv"
        frame = pd.read_csv(input_path, low_memory=False)
        required = {"_npy_path", "_evtype", "_evt6", "event_uid", "p_pick", "s_pick"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{input_path} is missing: {sorted(missing)}")

        p_pick = pd.to_numeric(frame["p_pick"], errors="coerce")
        s_pick = pd.to_numeric(frame["s_pick"], errors="coerce")
        valid = (
            p_pick.notna()
            & s_pick.notna()
            & (p_pick >= 0)
            & (s_pick > p_pick)
            & ((s_pick + args.post_s_samples) <= args.wave_length)
        )
        selected = frame.loc[valid].copy().reset_index(drop=True)
        if selected.empty:
            raise RuntimeError(f"S-eligible {split} split is empty")

        # The source paths are relative to src_dir. New metadata uses absolute
        # paths, so no waveform copies or symlinks are needed.
        selected["_npy_path"] = selected["_npy_path"].map(
            lambda value: as_absolute_path(value, src_dir)
        )
        selected["_s_eligible"] = 1
        selected["_sp_seconds"] = (
            (pd.to_numeric(selected["s_pick"], errors="coerce")
             - pd.to_numeric(selected["p_pick"], errors="coerce"))
            / 50.0
        ).astype(np.float32)
        output_path = dst_dir / f"meta_evt6_{split}.csv"
        selected.to_csv(output_path, index=False)

        event_sets[split] = set(selected["event_uid"].astype(str))
        report["splits"][split] = {
            "before": {
                "records": int(len(frame)),
                "events": int(frame["event_uid"].nunique()),
                "records_by_class": counts_by_class(frame),
                "events_by_class": event_counts_by_class(frame),
            },
            "after": {
                "records": int(len(selected)),
                "events": int(selected["event_uid"].nunique()),
                "records_by_class": counts_by_class(selected),
                "events_by_class": event_counts_by_class(selected),
            },
            "removed_records": int(len(frame) - len(selected)),
            "removed_events": int(frame["event_uid"].nunique() - selected["event_uid"].nunique()),
            "metadata": str(output_path),
        }

    overlap = {
        "train_val": len(event_sets["train"] & event_sets["val"]),
        "train_test": len(event_sets["train"] & event_sets["test"]),
        "val_test": len(event_sets["val"] & event_sets["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Event leakage in filtered protocol: {overlap}")
    report["event_overlap"] = overlap
    with (dst_dir / "s_eligible_protocol_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"Created S-eligible protocol: {dst_dir}")
    print(f"Event overlap: {overlap}")
    for split, values in report["splits"].items():
        print(
            f"{split}: records {values['before']['records']} -> {values['after']['records']}; "
            f"events {values['before']['events']} -> {values['after']['events']}"
        )


if __name__ == "__main__":
    main()

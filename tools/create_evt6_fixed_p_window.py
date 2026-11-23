#!/usr/bin/env python3
"""Crop an existing fixed-split EVT6 protocol to the CapsNet fixed-P window.

The output keeps the same rows and event splits. At 50 Hz, the default window
is [P-10 s, P+50 s), or 3000 samples. S picks are never used to choose a crop.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = ("train", "val", "test")


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dir", type=Path, required=True)
    parser.add_argument("--dst-dir", type=Path, required=True)
    parser.add_argument("--pre-p-samples", type=int, default=500)
    parser.add_argument("--post-p-samples", type=int, default=2500)
    return parser.parse_args()


def source_wave_path(root: Path, value: object) -> Path:
    path = Path(str(value).strip().replace("\\", "/"))
    return path if path.is_absolute() else root / path


def check_split_membership(frames: dict[str, pd.DataFrame]) -> None:
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        left_events = set(frames[left]["event_uid"].astype(str))
        right_events = set(frames[right]["event_uid"].astype(str))
        if left_events & right_events:
            raise ValueError(f"Event overlap between {left} and {right}")
        left_records = set(zip(frames[left]["key"].astype(str), frames[left]["part"].astype(str)))
        right_records = set(zip(frames[right]["key"].astype(str), frames[right]["part"].astype(str)))
        if left_records & right_records:
            raise ValueError(f"Record overlap between {left} and {right}")


def main() -> None:
    args = args_parser()
    src = args.src_dir.expanduser().resolve()
    dst = args.dst_dir.expanduser().resolve()
    pre = args.pre_p_samples
    post = args.post_p_samples
    length = pre + post
    if pre < 0 or post <= 0:
        raise ValueError("Expected nonnegative pre-P and positive post-P samples")
    if src == dst or src in dst.parents or dst in src.parents:
        raise ValueError("Source and destination must be separate directories")
    if not src.is_dir():
        raise FileNotFoundError(src)

    frames: dict[str, pd.DataFrame] = {}
    required = {"key", "part", "_evt6", "_npy_path", "event_uid", "p_pick"}
    for split in SPLITS:
        path = src / f"meta_evt6_{split}.csv"
        frame = pd.read_csv(path, low_memory=False)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if frame.duplicated(["key", "part"]).any():
            raise ValueError(f"Duplicate record identifiers in {path}")
        frames[split] = frame
    check_split_membership(frames)

    config = {
        "source_data_dir": str(src),
        "sample_rate_hz": 50,
        "pre_p_samples": pre,
        "post_p_samples": post,
        "window_samples": length,
        "interval": "[P-pre_p_samples, P+post_p_samples)",
        "s_pick_used_for_cropping": False,
    }
    dst.mkdir(parents=True, exist_ok=True)
    config_path = dst / "fixed_p_window_config.json"
    if config_path.exists():
        with config_path.open(encoding="utf-8") as handle:
            if json.load(handle) != config:
                raise ValueError(f"Existing destination has different settings: {dst}")
    elif any(dst.iterdir()):
        raise FileExistsError(f"Destination is nonempty and has no matching config: {dst}")
    else:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {"config": config, "splits": {}}
    for split, source_frame in frames.items():
        frame = source_frame.copy()
        output_paths = []
        s_inside_window = 0
        for index, row in source_frame.iterrows():
            p_value = pd.to_numeric(row["p_pick"], errors="coerce")
            if not np.isfinite(p_value):
                raise ValueError(f"{split} row {index}: invalid p_pick")
            p = int(round(float(p_value)))
            start = p - pre
            stop = p + post
            source_path = source_wave_path(src, row["_npy_path"])
            relative_path = Path("waves") / split / f"{index:08d}.npy"
            output_path = dst / relative_path
            output_paths.append(relative_path.as_posix())

            if not output_path.exists():
                wave = np.load(source_path, mmap_mode="r", allow_pickle=False)
                if wave.ndim != 2 or wave.shape[0] != 3 or start < 0 or stop > wave.shape[1]:
                    raise ValueError(
                        f"{split} row {index}: cannot crop {source_path}, "
                        f"shape={wave.shape}, requested=[{start}:{stop}]"
                    )
                crop = np.asarray(wave[:, start:stop], dtype=np.float32)
                if crop.shape != (3, length) or not np.isfinite(crop).all():
                    raise ValueError(f"{split} row {index}: invalid cropped waveform")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = output_path.with_suffix(".npy.tmp")
                with temp_path.open("wb") as handle:
                    np.save(handle, crop, allow_pickle=False)
                os.replace(temp_path, output_path)
            else:
                existing = np.load(output_path, mmap_mode="r", allow_pickle=False)
                if existing.shape != (3, length):
                    raise ValueError(f"Existing output has incorrect shape: {output_path}")

            frame.at[index, "p_pick"] = pre
            if "s_pick" in frame.columns:
                s_value = pd.to_numeric(row["s_pick"], errors="coerce")
                new_s = float(s_value) - start if np.isfinite(s_value) else np.nan
                frame.at[index, "s_pick"] = new_s
                s_inside_window += int(np.isfinite(new_s) and 0 <= new_s < length)
        frame["_npy_path"] = output_paths
        frame["_window_pre_p_samples"] = pre
        frame["_window_post_p_samples"] = post
        output_meta = dst / f"meta_evt6_{split}.csv"
        temp_meta = output_meta.with_suffix(".csv.tmp")
        frame.to_csv(temp_meta, index=False)
        os.replace(temp_meta, output_meta)
        report["splits"][split] = {
            "records": len(frame),
            "events": int(frame["event_uid"].nunique()),
            "class_counts": {str(k): int(v) for k, v in frame["_evt6"].value_counts().sort_index().items()},
            "s_pick_inside_window": s_inside_window,
        }
        print(f"{split}: {len(frame)} records, {frame['event_uid'].nunique()} events", flush=True)

    (dst / "fixed_p_window_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Completed fixed-P {length}-sample protocol: {dst}")


if __name__ == "__main__":
    main()

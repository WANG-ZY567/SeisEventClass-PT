#!/usr/bin/env python3
"""
Create a station-transfer protocol dir for EVT6 (train on seen stations, test on unseen stations).

Input requirements:
- src_dir contains meta_evt6.csv with columns: _evt6, station_id_raw, event_uid, _npy_path, ...
- waves paths can be absolute; this script can optionally symlink waves_non/ for completeness.

Outputs:
- dst_dir/meta_evt6.csv (symlink or copy)
- dst_dir/meta_evt6_{train,val,test}.csv (station-disjoint split)
- dst_dir/selection_record_evt6_station_transfer.json (split recipe + stats)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd


def _safe_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(src), str(dst))


def _safe_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


def _pick_station_sets(
    stations: np.ndarray,
    seed: int,
    test_frac: float,
    val_frac: float,
) -> Tuple[set, set, set]:
    rng = np.random.default_rng(int(seed))
    st = np.array(sorted(set(stations.tolist())), dtype=object)
    rng.shuffle(st)
    n = len(st)
    n_test = int(round(float(test_frac) * n))
    n_val = int(round(float(val_frac) * n))
    n_test = max(1, n_test) if n >= 3 else max(0, n_test)
    n_val = max(1, n_val) if n >= 3 else max(0, n_val)
    if n_test + n_val >= n:
        # keep at least 1 train station when possible
        n_val = max(0, n - n_test - 1)
        if n_val == 0 and n_test >= n:
            n_test = max(0, n - 1)

    test_st = set(st[:n_test].tolist())
    val_st = set(st[n_test : n_test + n_val].tolist())
    train_st = set(st[n_test + n_val :].tolist())
    return train_st, val_st, test_st


def _stats(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["rows"] = int(len(df))
    if "_evt6" in df.columns:
        out["counts_by_evt6"] = df["_evt6"].value_counts().sort_index().to_dict()
    if "station_id_raw" in df.columns:
        out["num_stations"] = int(df["station_id_raw"].astype(str).nunique())
    if "event_uid" in df.columns:
        out["num_events"] = int(df["event_uid"].astype(str).nunique())
    if "_src" in df.columns:
        out["counts_by_src"] = df["_src"].value_counts().to_dict()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True, help="Existing EVT6 dataset dir (xapp meta_evt6.csv)")
    ap.add_argument("--dst_dir", required=True, help="New protocol dir to create")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--test_station_frac", type=float, default=0.2)
    ap.add_argument("--val_station_frac", type=float, default=0.1)
    ap.add_argument("--link_mode", default="symlink", choices=["symlink", "copy"])
    args = ap.parse_args()

    src = Path(args.src_dir).expanduser().resolve()
    dst = Path(args.dst_dir).expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)

    src_meta = src / "meta_evt6.csv"
    if not src_meta.exists():
        raise FileNotFoundError(f"missing: {src_meta}")

    # reuse waves_non (optional)
    src_waves_non = src / "waves_non"
    if src_waves_non.exists():
        if args.link_mode == "symlink":
            _safe_symlink(src_waves_non, dst / "waves_non")
        else:
            if not (dst / "waves_non").exists():
                shutil.copytree(str(src_waves_non), str(dst / "waves_non"))

    # meta_evt6.csv
    if args.link_mode == "symlink":
        _safe_symlink(src_meta, dst / "meta_evt6.csv")
    else:
        _safe_copy(src_meta, dst / "meta_evt6.csv")

    df = pd.read_csv(str(src_meta), low_memory=False)
    for col in ("_evt6", "station_id_raw"):
        if col not in df.columns:
            raise ValueError(f"meta_evt6.csv missing required column: {col}")

    stations = df["station_id_raw"].astype(str).to_numpy()
    train_st, val_st, test_st = _pick_station_sets(
        stations=stations,
        seed=int(args.seed),
        test_frac=float(args.test_station_frac),
        val_frac=float(args.val_station_frac),
    )

    split = {}
    split["train"] = df[df["station_id_raw"].astype(str).isin(train_st)].copy()
    split["val"] = df[df["station_id_raw"].astype(str).isin(val_st)].copy()
    split["test"] = df[df["station_id_raw"].astype(str).isin(test_st)].copy()

    # deterministic shuffle within split (membership fixed)
    for s in ("train", "val", "test"):
        split[s] = split[s].sample(frac=1.0, random_state=int(args.seed)).reset_index(drop=True)
        split[s].to_csv(str(dst / f"meta_evt6_{s}.csv"), index=False)

    record: Dict[str, Any] = {
        "protocol": "station_transfer",
        "src_dir": str(src),
        "dst_dir": str(dst),
        "seed": int(args.seed),
        "test_station_frac": float(args.test_station_frac),
        "val_station_frac": float(args.val_station_frac),
        "stations": {
            "train": sorted(list(train_st))[:50],
            "val": sorted(list(val_st))[:50],
            "test": sorted(list(test_st))[:50],
            "num_train": int(len(train_st)),
            "num_val": int(len(val_st)),
            "num_test": int(len(test_st)),
            "note": "lists are truncated to first 50 for readability",
        },
        "stats": {s: _stats(split[s]) for s in ("train", "val", "test")},
    }
    with open(dst / "selection_record_evt6_station_transfer.json", "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print("[OK] created station-transfer protocol dir")
    print("src:", src)
    print("dst:", dst)
    print("stations:", {k: len(v) for k, v in (("train", train_st), ("val", val_st), ("test", test_st))})
    print("stats:", {k: record["stats"][k]["rows"] for k in ("train", "val", "test")})


if __name__ == "__main__":
    main()


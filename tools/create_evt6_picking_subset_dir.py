#!/usr/bin/env python3
"""
Create a picking-ready subset protocol dir from an EVT6 xapp dataset directory.

Goal:
- Keep only samples with usable p_pick/s_pick (optionally require both P and S).
- Reuse existing meta_evt6_{train,val,test}.csv split membership (by default),
  or derive splits from meta_evt6.csv if requested.

This is a data-prep step for "class-conditioned phase picking" experiments.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List

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


def _to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _filter_picks(df: pd.DataFrame, require_s: bool, in_samples: int) -> pd.DataFrame:
    if "p_pick" not in df.columns:
        raise ValueError("missing column: p_pick")
    df = df.copy()
    df["p_pick"] = _to_float_series(df["p_pick"])
    if "s_pick" in df.columns:
        df["s_pick"] = _to_float_series(df["s_pick"])
    else:
        df["s_pick"] = np.nan

    # Basic bounds in cropped window coords: [0, in_samples)
    m_p = df["p_pick"].notna() & (df["p_pick"] >= 0) & (df["p_pick"] < float(in_samples))
    if require_s:
        m_s = df["s_pick"].notna() & (df["s_pick"] >= 0) & (df["s_pick"] < float(in_samples))
        m = m_p & m_s
    else:
        m = m_p
    return df.loc[m].reset_index(drop=True)


def _stats(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {"rows": int(len(df))}
    if "_evt6" in df.columns:
        out["counts_by_evt6"] = df["_evt6"].value_counts().sort_index().to_dict()
    if "station_id_raw" in df.columns:
        out["num_stations"] = int(df["station_id_raw"].astype(str).nunique())
    if "event_uid" in df.columns:
        out["num_events"] = int(df["event_uid"].astype(str).nunique())
    if "_src" in df.columns:
        out["counts_by_src"] = df["_src"].value_counts().to_dict()
    # pick availability stats
    if "p_pick" in df.columns:
        out["p_pick_na"] = int(pd.to_numeric(df["p_pick"], errors="coerce").isna().sum())
    if "s_pick" in df.columns:
        out["s_pick_na"] = int(pd.to_numeric(df["s_pick"], errors="coerce").isna().sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True, help="EVT6 xapp dir (has meta_evt6_{train,val,test}.csv)")
    ap.add_argument("--dst_dir", required=True, help="New dir to create (picking subset)")
    ap.add_argument("--in_samples", type=int, default=8192)
    ap.add_argument(
        "--require_s",
        action="store_true",
        help="Only keep samples with both p_pick and s_pick in [0,in_samples).",
    )
    ap.add_argument("--link_mode", default="symlink", choices=["symlink", "copy"])
    args = ap.parse_args()

    src = Path(args.src_dir).expanduser().resolve()
    dst = Path(args.dst_dir).expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)

    # reuse waves_non (optional)
    src_waves_non = src / "waves_non"
    if src_waves_non.exists():
        if args.link_mode == "symlink":
            _safe_symlink(src_waves_non, dst / "waves_non")
        else:
            if not (dst / "waves_non").exists():
                shutil.copytree(str(src_waves_non), str(dst / "waves_non"))

    # meta_evt6.csv
    src_meta = src / "meta_evt6.csv"
    if not src_meta.exists():
        raise FileNotFoundError(f"missing: {src_meta}")
    if args.link_mode == "symlink":
        _safe_symlink(src_meta, dst / "meta_evt6.csv")
    else:
        _safe_copy(src_meta, dst / "meta_evt6.csv")

    record: Dict[str, Any] = {
        "protocol": "picking_subset",
        "src_dir": str(src),
        "dst_dir": str(dst),
        "in_samples": int(args.in_samples),
        "require_s": bool(args.require_s),
        "filter": {
            "p_pick": "[0,in_samples)",
            "s_pick": "[0,in_samples)" if bool(args.require_s) else "optional",
        },
        "stats": {},
    }

    for split in ("train", "val", "test"):
        p = src / f"meta_evt6_{split}.csv"
        if not p.exists():
            raise FileNotFoundError(f"missing: {p}")
        df = pd.read_csv(str(p), low_memory=False)
        before = _stats(df)
        df2 = _filter_picks(df, require_s=bool(args.require_s), in_samples=int(args.in_samples))
        after = _stats(df2)
        record["stats"][split] = {"before": before, "after": after}
        df2.to_csv(str(dst / f"meta_evt6_{split}.csv"), index=False)

    with open(dst / "selection_record_evt6_picking_subset.json", "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print("[OK] created picking subset dir")
    print("src:", src)
    print("dst:", dst)
    print("require_s:", bool(args.require_s))
    for split in ("train", "val", "test"):
        b = record["stats"][split]["before"]["rows"]
        a = record["stats"][split]["after"]["rows"]
        print(f"{split}: {a}/{b}")


if __name__ == "__main__":
    main()


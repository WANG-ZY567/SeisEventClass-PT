#!/usr/bin/env python3
"""
data EVT6 xapp picking subset data"data evt6 data"data. 

value: 
- Oracle-conditioned picking: data evt6 data picker(data). 

value: 
- src_dir: data meta_evt6_{train,val,test}.csv(value: _evt6)

value: 
- dst_root/<class_name or id>/meta_evt6_{train,val,test}.csv
- data meta_evt6.csv data waves_non(data)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

ID2NAME = {0: "eq", 1: "ep", 2: "co", 3: "sp", 4: "se", 5: "ot"}


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True)
    ap.add_argument("--dst_root", required=True)
    ap.add_argument(
        "--classes",
        default="0,1,2,3,4,5",
        help="data class id data(0-5), data",
    )
    ap.add_argument("--link_mode", default="symlink", choices=["symlink", "copy"])
    args = ap.parse_args()

    src = Path(args.src_dir).expanduser().resolve()
    dst_root = Path(args.dst_root).expanduser().resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    class_ids: List[int] = []
    for p in [x.strip() for x in str(args.classes).split(",") if x.strip()]:
        class_ids.append(int(p))

    # link/copy common assets once per class dir
    src_meta = src / "meta_evt6.csv"
    if not src_meta.exists():
        raise FileNotFoundError(f"missing: {src_meta}")

    src_waves_non = src / "waves_non"

    record: Dict[str, Any] = {
        "protocol": "evt6_class_subset",
        "src_dir": str(src),
        "dst_root": str(dst_root),
        "classes": class_ids,
        "stats": {},
    }

    for cid in class_ids:
        cname = ID2NAME.get(cid, str(cid))
        dst = dst_root / f"evt6_{cid}_{cname}"
        dst.mkdir(parents=True, exist_ok=True)

        # link/copy meta_evt6.csv
        if args.link_mode == "symlink":
            _safe_symlink(src_meta, dst / "meta_evt6.csv")
        else:
            _safe_copy(src_meta, dst / "meta_evt6.csv")

        # link/copy waves_non if exists
        if src_waves_non.exists():
            if args.link_mode == "symlink":
                _safe_symlink(src_waves_non, dst / "waves_non")
            else:
                if not (dst / "waves_non").exists():
                    shutil.copytree(str(src_waves_non), str(dst / "waves_non"))

        record["stats"][str(cid)] = {}
        for split in ("train", "val", "test"):
            p = src / f"meta_evt6_{split}.csv"
            if not p.exists():
                raise FileNotFoundError(f"missing: {p}")
            df = pd.read_csv(str(p), low_memory=False)
            if "_evt6" not in df.columns:
                raise KeyError(f"missing column _evt6 in {p}")
            before = int(df.shape[0])
            df2 = df.loc[df["_evt6"].astype(int) == int(cid)].reset_index(drop=True)
            after = int(df2.shape[0])
            df2.to_csv(str(dst / f"meta_evt6_{split}.csv"), index=False)
            record["stats"][str(cid)][split] = {"before": before, "after": after}

    with open(dst_root / "selection_record_evt6_class_subset.json", "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print("[OK] created class subset dirs under:", dst_root)


if __name__ == "__main__":
    main()


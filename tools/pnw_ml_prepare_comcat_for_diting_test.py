#!/usr/bin/env python3
"""
从 ComCat metadata 生成可供 `dataset-name diting2_evt6` 使用的 `meta_evt6_test.csv`。

列约定（与 DiTing2Evt6 读取逻辑一致）：
  - key, part, _npy_path, _evt6

标签对齐（用于与 DiTing EVT6 六类 head 对照）：
  - ComCat earthquake -> _evt6=0  （对应 DiTing evt6: eq）
  - ComCat explosion  -> _evt6=1  （对应 DiTing evt6: ep）

_npy_path 约定为相对 data_dir 的路径，例如 waves/<safe_key>.npy
需先用 pnw_ml_export_hdf5_to_npy.py 导出 npy。

可选列（透传，便于追溯）：
  comcat_event_id, station_code, trace_name
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)[:200]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comcat_csv", type=Path, default=Path("/path/to/comcat_metadata.csv"))
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument(
        "--waves_subdir",
        default="waves",
        help="相对于 external data_dir 的子目录名",
    )
    args = ap.parse_args()

    rows_out = []
    with args.comcat_csv.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            st = (row.get("source_type") or "").strip().lower()
            if st not in ("earthquake", "explosion"):
                continue
            tid = (row.get("trace_name") or "").strip()
            eid = (row.get("event_id") or "").strip()
            st_code = (row.get("station_code") or "").strip()
            if not tid:
                continue
            cls_id = 0 if st == "earthquake" else 1
            key = f"{eid}|{st_code}|{tid}"
            npy_rel = f"{args.waves_subdir}/{safe_name(key)}.npy"
            rows_out.append(
                {
                    "key": key,
                    "part": "test",
                    "_npy_path": npy_rel,
                    "_evt6": cls_id,
                    "comcat_event_id": eid,
                    "station_code": st_code,
                    "trace_name": tid,
                }
            )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "key",
        "part",
        "_npy_path",
        "_evt6",
        "comcat_event_id",
        "station_code",
        "trace_name",
    ]
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)
    print("wrote", len(rows_out), "rows ->", args.out_csv)


if __name__ == "__main__":
    main()

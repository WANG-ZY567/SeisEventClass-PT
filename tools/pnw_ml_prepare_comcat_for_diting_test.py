#!/usr/bin/env python3
"""
data ComCat metadata `dataset-name diting2_evt6` data `meta_evt6_test.csv`. 

data(data DiTing2Evt6 data): 
  - key, part, _npy_path, _evt6

data(data DiTing EVT6 data head data): 
  - ComCat earthquake -> _evt6=0  (data DiTing evt6: eq)
  - ComCat explosion  -> _evt6=1  (data DiTing evt6: ep)

_npy_path is stored relative to data_dir, for example waves/<safe_key>.npy
data pnw_ml_export_hdf5_to_npy.py data npy. 

data(data, data): 
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
        help="data external data_dir data",
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


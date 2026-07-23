#!/usr/bin/env python3
"""
data ComCat / Exotic data HDF5 data `meta_evt6_test.csv`(data `dataset` + `trace_name`)data npy. 

value: data `pnw_ml_export_hdf5_to_npy.py` data `read_waveform_from_h5` / `to_3c_L`. 
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from pnw_ml_export_hdf5_to_npy import read_waveform_from_h5, to_3c_L  # noqa: E402


def main() -> None:
    try:
        import h5py  # type: ignore
        import numpy as np
    except ImportError:
        print("value: pip install h5py numpy")
        raise SystemExit(1)

    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5_comcat", type=Path, required=True)
    ap.add_argument("--hdf5_exotic", type=Path, required=True)
    ap.add_argument("--meta_csv", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--in_samples", type=int, default=8192)
    ap.add_argument("--max_rows", type=int, default=0, help="0 data")
    args = ap.parse_args()

    rows: list[dict] = []
    with args.meta_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    n_ok = 0
    n_fail = 0
    with h5py.File(args.hdf5_comcat, "r") as h5_c, h5py.File(args.hdf5_exotic, "r") as h5_e:
        for row in rows:
            tn = (row.get("trace_name") or "").strip()
            rel = (row.get("_npy_path") or "").strip()
            ds = (row.get("dataset") or "").strip().lower()
            if not tn or not rel or ds not in ("comcat", "exotic"):
                n_fail += 1
                continue
            h5 = h5_c if ds == "comcat" else h5_e
            out_path = args.data_dir / rel
            try:
                raw = read_waveform_from_h5(h5, tn)
                arr = to_3c_L(raw, args.in_samples)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(out_path, arr)
                n_ok += 1
            except Exception as e:
                n_fail += 1
                if n_fail <= 5:
                    print("FAIL", ds, tn, e)

    print("ok", n_ok, "fail", n_fail)


if __name__ == "__main__":
    main()

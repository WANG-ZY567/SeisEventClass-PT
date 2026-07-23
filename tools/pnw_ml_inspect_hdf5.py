#!/usr/bin/env python3
"""
data HDF5 dataset data(data h5py). 
data. data trace_name data. 
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    try:
        import h5py  # type: ignore
    except ImportError:
        print("value: pip install h5py")
        raise SystemExit(1)

    ap = argparse.ArgumentParser()
    ap.add_argument("hdf5_path", type=Path)
    ap.add_argument("--max_datasets", type=int, default=30, help="dataset")
    args = ap.parse_args()

    path = args.hdf5_path
    if not path.is_file():
        print("value:", path)
        raise SystemExit(1)

    n = 0
    with h5py.File(path, "r") as f:
        print("value:", path)
        print("data keys (data 50):", list(f.keys())[:50])

        def visitor(name, obj):
            nonlocal n
            if isinstance(obj, h5py.Dataset) and n < args.max_datasets:
                print(f"  [D] {name} shape={obj.shape} dtype={obj.dtype}")
                n += 1

        f.visititems(visitor)
        if n >= args.max_datasets:
            print(f"... data {args.max_datasets} dataset, data --max_datasets")


if __name__ == "__main__":
    main()

"""Inspect whether an HDF5 file can be opened by h5py.

Example:
    python tools/check_hdf5.py --h5 /path/to/CENC_DiTingv2_natural_earthquake.hdf5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def try_open(path: Path, title: str, **kwargs) -> None:
    print(f"\n{title}")
    try:
        with h5py.File(path, "r", **kwargs) as handle:
            keys = list(handle.keys())
            print("  status: ok")
            print(f"  root attrs: {dict(handle.attrs)}")
            print(f"  num keys: {len(keys)}")
            print(f"  first keys: {keys[:5]}")
    except Exception as exc:  # noqa: BLE001 - diagnostic script should report all failures.
        print("  status: failed")
        print(f"  error: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check basic HDF5 readability.")
    parser.add_argument("--h5", required=True, type=Path, help="Path to the HDF5 file.")
    args = parser.parse_args()

    if not args.h5.exists():
        raise FileNotFoundError(args.h5)

    print(f"h5py version: {h5py.version.version}")
    print(f"HDF5 library version: {h5py.version.hdf5_version}")
    print(f"file: {args.h5}")

    try_open(args.h5, "Method 1: default")
    try_open(args.h5, "Method 2: SWMR", swmr=True)
    try_open(args.h5, "Method 3: libver='latest'", libver="latest")
    try_open(args.h5, "Method 4: larger raw data chunk cache", rdcc_nbytes=1024**3)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Open-source note: implementation detail."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=Path, required=True)
    ap.add_argument(
        "--splits",
        required=True,
        help="data, data source_train,source_val",
    )
    ap.add_argument("--out_csv", type=Path, required=True)
    args = ap.parse_args()
    want = {s.strip() for s in args.splits.split(",") if s.strip()}
    rows = []
    with args.in_csv.open(newline="") as f:
        r = csv.DictReader(f)
        fields = r.fieldnames
        for row in r:
            if (row.get("split") or "").strip() in want:
                rows.append(row)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or [])
        w.writeheader()
        w.writerows(rows)
    print("kept", len(rows), "rows ->", args.out_csv)


if __name__ == "__main__":
    main()

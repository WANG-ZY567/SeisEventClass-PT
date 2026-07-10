#!/usr/bin/env python3
"""
ComCat + Exotic：生成 4 类 trace 级 manifest（扩展实验）。

类别定义（方案 B，推荐）：
  0 earthquake   <- ComCat source_type==earthquake
  1 explosion    <- ComCat source_type==explosion
  2 surface_event <- Exotic source_type==surface event
  3 other_exotic  <- Exotic: sonic boom + thunder + plane crash

说明：
  - Exotic 无震源经纬度；若要做「与 ComCat 一致的 region split」，可用 station 坐标近似，
    或仅将 Exotic 作为「类型扩展」子集，在单独协议中报告（见主文档）。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--comcat_csv",
        type=Path,
        default=Path("/path/to/comcat_metadata.csv"),
    )
    ap.add_argument(
        "--exotic_csv",
        type=Path,
        default=Path("/path/to/exotic_metadata.csv"),
    )
    ap.add_argument("--out_csv", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    with args.comcat_csv.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            st = (row.get("source_type") or "").strip().lower()
            if st == "earthquake":
                lab = 0
            elif st == "explosion":
                lab = 1
            else:
                continue
            rows.append(
                {
                    "dataset": "comcat",
                    "event_id": row.get("event_id", ""),
                    "trace_name": row.get("trace_name", ""),
                    "label": lab,
                    "label_name": st,
                }
            )

    other_map = {
        "sonic boom": 3,
        "thunder": 3,
        "plane crash": 3,
        "surface event": 2,
    }
    with args.exotic_csv.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            st = (row.get("source_type") or "").strip()
            if st not in other_map:
                continue
            lab = other_map[st]
            lname = "surface_event" if lab == 2 else "other_exotic"
            rows.append(
                {
                    "dataset": "exotic",
                    "event_id": row.get("event_id", ""),
                    "trace_name": row.get("trace_name", ""),
                    "label": lab,
                    "label_name": lname,
                }
            )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "event_id", "trace_name", "label", "label_name"]
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("wrote", len(rows), "rows ->", args.out_csv)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Exotic：生成 meta_evt6_test.csv，用于 **仅推理**（OOD / case study）。

说明：
  - DiTing EVT6 与 Exotic **不对齐**；默认 `--dummy_evt6 0` 仅为满足部分测试管线对标签张量的形状要求。
  - **不要** 用该 dummy 标签计算「DiTing 原任务准确率」；Exotic 只看 **pred_evt6 分布** 与 case study。

附加列 `pnw_source_type` 供后续统计预测分布。

导出 npy 后路径与 ComCat 脚本同约定。
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
    ap.add_argument("--exotic_csv", type=Path, default=Path("/path/to/exotic_metadata.csv"))
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--waves_subdir", default="waves")
    ap.add_argument(
        "--dummy_evt6",
        type=int,
        default=0,
        help="占位标签 0..5；Exotic 分析请忽略 tgt，只看预测分布",
    )
    args = ap.parse_args()

    rows_out = []
    with args.exotic_csv.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            tid = (row.get("trace_name") or "").strip()
            eid = (row.get("event_id") or "").strip()
            st_code = (row.get("station_code") or "").strip()
            ptype = (row.get("source_type") or "").strip()
            if not tid:
                continue
            key = f"{eid}|{st_code}|{tid}"
            npy_rel = f"{args.waves_subdir}/{safe_name(key)}.npy"
            rows_out.append(
                {
                    "key": key,
                    "part": "test",
                    "_npy_path": npy_rel,
                    "_evt6": args.dummy_evt6,
                    "pnw_source_type": ptype,
                    "exotic_event_id": eid,
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
        "pnw_source_type",
        "exotic_event_id",
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

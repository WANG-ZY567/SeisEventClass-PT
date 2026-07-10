#!/usr/bin/env python3
"""
从 `pnw_ml_build_multiclass_manifest.py` 生成的 manifest（或等价列）生成
`diting2_evt6` 可用的 `meta_evt6_test.csv`。

PNW 4 类（写入 _evt6，与 DiTing 六类 id **不同语义**）：
  0 earthquake, 1 explosion, 2 surface_event, 3 other_exotic

说明：外测时模型仍为 **6 类 softmax**；`_evt6` 仅作 **PNW 真值** 写入 test_results，
评测请用 `pnw_ml_eval_pnw4_multiclass.py`（4×6 混淆等），勿与 DiTing 内测六类直接混比。
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
    ap.add_argument("--manifest_csv", type=Path, required=True, help="multiclass manifest（含 dataset,event_id,trace_name,label,label_name）")
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--waves_subdir", default="waves")
    args = ap.parse_args()

    rows_out = []
    with args.manifest_csv.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ds = (row.get("dataset") or "").strip()
            tid = (row.get("trace_name") or "").strip()
            eid = (row.get("event_id") or "").strip()
            lab = row.get("label", "")
            lname = (row.get("label_name") or "").strip()
            if not tid or ds not in ("comcat", "exotic"):
                continue
            try:
                cls_id = int(lab)
            except Exception:
                continue
            if cls_id < 0 or cls_id > 3:
                continue
            key = f"{ds}|{eid}|{tid}"
            npy_rel = f"{args.waves_subdir}/{safe_name(key)}.npy"
            rows_out.append(
                {
                    "key": key,
                    "part": "test",
                    "_npy_path": npy_rel,
                    "_evt6": cls_id,
                    "dataset": ds,
                    "pnw_event_id": eid,
                    "trace_name": tid,
                    "label_name": lname,
                }
            )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["key", "part", "_npy_path", "_evt6", "dataset", "pnw_event_id", "trace_name", "label_name"]
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)
    print("wrote", len(rows_out), "rows ->", args.out_csv)


if __name__ == "__main__":
    main()

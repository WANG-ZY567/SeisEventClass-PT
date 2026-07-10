#!/usr/bin/env python3
"""
EVT6 class sensitivity analysis over stratified bins (magnitude / distance).

Goal (paper-oriented):
- Quantify how each EVT6 class's performance changes with mag/dist.
- Avoid focusing on model-vs-model; instead highlight class-conditional robustness.

Inputs:
- `strat_<field>_<scheme>_per_class_f1.csv` produced by `tools/evt6_stratified_analysis.py`
  columns: bin_id, bin_label, class_id, class_name, support, precision, recall, f1

Outputs (in --out_dir):
- `class_sensitivity_<field>_<scheme>.csv` (per-class summary stats)
- `CLASS_SENSITIVITY_REPORT.md` (paper-friendly notes + tables)

This script is dependency-light: uses only Python stdlib.
"""

from __future__ import annotations

import argparse
import csv
import os
import math
from typing import Dict, List, Tuple


EVT6_ID2NAME = {0: "eq", 1: "ep", 2: "co", 3: "sp", 4: "se", 5: "ot"}
CLASSES = list(range(6))


def _read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [row for row in r]


def _safe_float(x: str, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default


def _safe_int(x: str, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def _weighted_mean(values: List[float], weights: List[float]) -> float:
    s = sum(weights)
    if s <= 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / s


def _pearson(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    vx = sum((xi - mx) ** 2 for xi in x)
    vy = sum((yi - my) ** 2 for yi in y)
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    return cov / math.sqrt(vx * vy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_class_csv", required=True, help="strat_<field>_<scheme>_per_class_f1.csv path")
    ap.add_argument("--out_dir", required=True, help="output directory")
    ap.add_argument("--field", required=True, help="field name (mag / Pg_dist / etc) for labeling only")
    ap.add_argument("--scheme", required=True, help="scheme tag (quantile5 / fixed) for labeling only")
    args = ap.parse_args()

    per_class_csv = os.path.abspath(args.per_class_csv)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows = _read_rows(per_class_csv)
    if not rows:
        raise RuntimeError(f"empty per_class_csv: {per_class_csv}")

    # group by class
    by_class: Dict[int, List[Dict[str, str]]] = {k: [] for k in CLASSES}
    bin_labels: Dict[int, str] = {}
    for r in rows:
        cid = _safe_int(r.get("class_id", ""), default=-1)
        if cid not in by_class:
            continue
        bid = _safe_int(r.get("bin_id", ""), default=-1)
        if bid >= 0:
            bin_labels[bid] = str(r.get("bin_label", ""))
        by_class[cid].append(r)

    # ensure bin order
    all_bins = sorted(bin_labels.keys())

    out_rows = []
    for cid in CLASSES:
        items = sorted(by_class[cid], key=lambda r: _safe_int(r.get("bin_id", ""), 0))
        f1s = [_safe_float(r.get("f1", "0")) for r in items]
        sups = [_safe_float(r.get("support", "0")) for r in items]

        if not items:
            continue

        # sensitivity metrics
        f1_min = min(f1s) if f1s else 0.0
        f1_max = max(f1s) if f1s else 0.0
        f1_range = f1_max - f1_min
        f1_wmean = _weighted_mean(f1s, sups)

        # correlation with bin id (monotonic trend proxy)
        x = list(range(len(f1s)))
        corr = _pearson(x, f1s)

        # end-to-end drop (first -> last)
        f1_first = f1s[0] if f1s else 0.0
        f1_last = f1s[-1] if f1s else 0.0
        delta_last_first = f1_last - f1_first

        out_rows.append(
            {
                "field": str(args.field),
                "scheme": str(args.scheme),
                "class_id": cid,
                "class_name": EVT6_ID2NAME.get(cid, str(cid)),
                "total_support": int(sum(sups)),
                "bins": len(f1s),
                "f1_weighted_mean": round(f1_wmean, 6),
                "f1_min": round(f1_min, 6),
                "f1_max": round(f1_max, 6),
                "f1_range": round(f1_range, 6),
                "f1_first": round(f1_first, 6),
                "f1_last": round(f1_last, 6),
                "f1_last_minus_first": round(delta_last_first, 6),
                "corr_binid_f1": round(corr, 6),
            }
        )

    out_csv = os.path.join(out_dir, f"class_sensitivity_{args.field}_{args.scheme}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(out_rows[0].keys()) if out_rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    md = os.path.join(out_dir, "CLASS_SENSITIVITY_REPORT.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("# EVT6 Class Sensitivity Report\n\n")
        f.write(f"- per_class_csv: `{per_class_csv}`\n")
        f.write(f"- field: `{args.field}`\n")
        f.write(f"- scheme: `{args.scheme}`\n\n")

        f.write("## How to read\n\n")
        f.write("- `f1_range` 越大：该类在不同震级/距离桶之间波动越大（更“敏感”）。\n")
        f.write("- `f1_last_minus_first`：最后一个桶相对第一个桶的变化方向（正/负）。\n")
        f.write("- `corr_binid_f1`：F1 随桶序号的线性趋势（仅作趋势提示，不作严格统计推断）。\n\n")

        f.write("## Bin labels\n\n")
        for bid in sorted(bin_labels.keys()):
            f.write(f"- bin {bid}: {bin_labels[bid]}\n")
        f.write("\n")

        f.write("## Per-class summary (CSV)\n\n")
        f.write(f"See `{os.path.basename(out_csv)}`.\n\n")

        f.write("## Paper-facing conservative notes template\n\n")
        f.write("- 报告每类在各桶的 support；对 support 很小的桶谨慎解释。\n")
        f.write("- 优先用 `f1_range` + per-bin 曲线描述“敏感程度”，避免过度因果化。\n")

    print("[OK] wrote:")
    print(" -", out_csv)
    print(" -", md)


if __name__ == "__main__":
    main()


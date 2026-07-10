#!/usr/bin/env python3
"""
Exotic case study：合并 `meta_evt6_test.csv`（含 pnw_source_type）与 `test_results_*.csv`（含 pred_evt6），
按 source_type 统计 pred_evt6 直方图与占比。

若 test_results 无 pnw_source_type，可用 --meta_csv join key=key。
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def load_meta_map(path: Path) -> dict[str, str]:
    m = {}
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            k = (row.get("key") or "").strip()
            if k:
                m[k] = (row.get("pnw_source_type") or "").strip()
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_csv", type=Path, required=True)
    ap.add_argument("--meta_csv", type=Path, default=None, help="含 key 与 pnw_source_type")
    ap.add_argument("--out_json", type=Path, required=True)
    args = ap.parse_args()

    meta_map = load_meta_map(args.meta_csv) if args.meta_csv else {}

    # results may have key column - check
    by_st: dict[str, list[int]] = defaultdict(list)
    pred_only = Counter()

    with args.results_csv.open(newline="") as f:
        r = csv.DictReader(f)
        fields = r.fieldnames or []
        has_key = "key" in fields
        for row in r:
            pr = int(float(row["pred_evt6"]))
            pred_only[pr] += 1
            st = (row.get("pnw_source_type") or "").strip()
            if not st and has_key:
                st = meta_map.get((row.get("key") or "").strip(), "unknown")
            if not st:
                st = "unknown"
            by_st[st].append(pr)

    out = {"by_source_type": {}, "pred_marginal": dict(pred_only)}
    EVT6_NAMES = {0: "eq", 1: "ep", 2: "co_ss", 3: "sp", 4: "se", 5: "ot"}
    for st, preds in sorted(by_st.items()):
        c = Counter(preds)
        out["by_source_type"][st] = {
            "n": len(preds),
            "pred_evt6_counts": {EVT6_NAMES.get(k, str(k)): v for k, v in sorted(c.items())},
        }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

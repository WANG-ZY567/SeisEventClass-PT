#!/usr/bin/env python3
"""
data target_train_pool data event data few-shot data(event data). 
value: fewshot_events.csv(event_id data), data traces data join. 
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def load_event_ids(path: Path) -> list[str]:
    out = []
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            eid = (row.get("event_id") or "").strip()
            if eid:
                out.append(eid)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--target_train_pool_csv",
        type=Path,
        required=True,
        help="events_target_train_pool.csv",
    )
    ap.add_argument("--frac", type=float, required=True, help="0.01 / 0.05 / 0.10")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", type=Path, required=True)
    args = ap.parse_args()

    eids = load_event_ids(args.target_train_pool_csv)
    rng = random.Random(args.seed)
    rng.shuffle(eids)
    k = max(1, int(round(len(eids) * args.frac)))
    picked = eids[:k]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "fewshot_frac", "seed"])
        for e in picked:
            w.writerow([e, args.frac, args.seed])
    print("pool events:", len(eids), "picked:", len(picked), "frac:", args.frac)
    print("wrote", args.out_csv)


if __name__ == "__main__":
    main()

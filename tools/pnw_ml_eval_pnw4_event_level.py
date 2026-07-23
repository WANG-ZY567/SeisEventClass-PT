#!/usr/bin/env python3
"""
PNW4 data(data 4 data, data 6 data). 

- value: data `dataset` + `pnw_event_id` data, data event_id data. 
- value: event data `tgt_evt6` data(mode). 
- value: event data `pred_evt6` data(mode). 

value: 
- data 4x6 contingency(true_pnw4 x pred_evt6)
- eq/ep data(true in {0,1} data pred==true)
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


PNW4 = ("earthquake", "explosion", "surface_event", "other_exotic")
EVT6 = ("eq", "ep", "co", "sp", "se", "ot")


def mode(xs: list[int]) -> int:
    return Counter(xs).most_common(1)[0][0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_csv", type=Path, required=True)
    ap.add_argument("--out_json", type=Path, required=True)
    ap.add_argument("--dataset_col", default="dataset")
    ap.add_argument("--event_id_col", default="pnw_event_id")
    args = ap.parse_args()

    by_event: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"t": [], "p": []})
    with args.results_csv.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ds = (row.get(args.dataset_col) or "").strip()
            eid = (row.get(args.event_id_col) or "").strip()
            if not ds or not eid:
                continue
            key = f"{ds}|{eid}"
            try:
                t = int(float(row["tgt_evt6"]))
                p = int(float(row["pred_evt6"]))
            except Exception:
                continue
            by_event[key]["t"].append(t)
            by_event[key]["p"].append(p)

    cm = [[0] * 6 for _ in range(4)]
    n_eq_ep = 0
    correct_eq_ep = 0
    total_events = 0

    for _, d in by_event.items():
        yt = mode(d["t"])
        yp = mode(d["p"])
        if 0 <= yt <= 3 and 0 <= yp <= 5:
            total_events += 1
            cm[yt][yp] += 1
            if yt in (0, 1):
                n_eq_ep += 1
                if yp == yt:
                    correct_eq_ep += 1

    acc_eq_ep = correct_eq_ep / n_eq_ep if n_eq_ep else 0.0
    row_sum = [sum(cm[i]) for i in range(4)]
    row_norm = [
        [cm[i][j] / row_sum[i] if row_sum[i] else 0.0 for j in range(6)]
        for i in range(4)
    ]

    report = {
        "n_events_used": total_events,
        "subset_event_accuracy_eq_ep_pred_matches_di_ting_eq_ep": acc_eq_ep,
        "n_subset_events_eq_ep": n_eq_ep,
        "note_eq_ep": "data true_event in {0,1}; data pred_event data true_event data. ",
        "contingency_true_pnw4_by_pred_evt6_event_level": {
            "true_pnw4_labels": list(PNW4),
            "pred_evt6_labels": list(EVT6),
            "counts_4x6": cm,
        },
        "row_normalized_4x6_event_level": row_norm,
        "support_per_true_pnw4_event_level": {PNW4[i]: row_sum[i] for i in range(4)},
        "note": "PNW4 data EVT6 data; 2/3 data. ",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

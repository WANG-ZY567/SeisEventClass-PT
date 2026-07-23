#!/usr/bin/env python3
"""
PNW 4 data(tgt_evt6 in {0,1,2,3})x DiTing data(pred_evt6 in {0..5})data. 

value: 
  - contingency_4x6: data PNW4 x data EVT6 data
  - subset_accuracy_eq_ep: data ComCat data(tgt 0/1)data pred==tgt data
  - row_normalized_4x6: data(data)
  - per_true_class_support: data

value: tgt data 2/3 data pred data 2..5 **data**, data"pred==tgt"data tgtin{2,3} interpret data. 
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PNW4 = ("earthquake", "explosion", "surface_event", "other_exotic")
EVT6 = ("eq", "ep", "co", "sp", "se", "ot")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_csv", type=Path, required=True)
    ap.add_argument("--out_json", type=Path, required=True)
    args = ap.parse_args()

    cm = [[0] * 6 for _ in range(4)]
    n_eq_ep = 0
    correct_eq_ep = 0
    n = 0

    with args.results_csv.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                yt = int(float(row["tgt_evt6"]))
                pr = int(float(row["pred_evt6"]))
            except Exception:
                continue
            if yt < 0 or yt > 3 or pr < 0 or pr > 5:
                continue
            n += 1
            cm[yt][pr] += 1
            if yt in (0, 1):
                n_eq_ep += 1
                if pr == yt:
                    correct_eq_ep += 1

    acc_ep = correct_eq_ep / n_eq_ep if n_eq_ep else 0.0
    row_sum = [sum(cm[i]) for i in range(4)]
    row_norm = [
        [cm[i][j] / row_sum[i] if row_sum[i] else 0.0 for j in range(6)]
        for i in range(4)
    ]

    report = {
        "n_rows_used": n,
        "subset_accuracy_eq_ep_pred_matches_di_ting_eq_ep": acc_ep,
        "n_subset_eq_ep": n_eq_ep,
        "note_eq_ep": "data tgtin{0,1}; data pred_evt6 data tgt data(data ComCat data). ",
        "contingency_true_pnw4_by_pred_evt6": {
            "true_pnw4_labels": list(PNW4),
            "pred_evt6_labels": list(EVT6),
            "counts_4x6": cm,
        },
        "row_normalized_4x6": row_norm,
        "support_per_true_pnw4": {PNW4[i]: row_sum[i] for i in range(4)},
        "note": "Rows are true PNW4 labels and columns are predicted EVT6 labels; off-diagonal cells indicate cross-label assignments.",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

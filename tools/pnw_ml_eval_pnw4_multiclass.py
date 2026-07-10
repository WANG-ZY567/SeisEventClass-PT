#!/usr/bin/env python3
"""
PNW 4 类真值（tgt_evt6 ∈ {0,1,2,3}）× DiTing 六类预测（pred_evt6 ∈ {0..5}）评测。

输出：
  - contingency_4x6：真值 PNW4 × 预测 EVT6 计数
  - subset_accuracy_eq_ep：仅 ComCat 语义（tgt 0/1）上 pred==tgt 的比例
  - row_normalized_4x6：按真值行归一化（便于看图）
  - per_true_class_support：各类真值样本数

注意：tgt 的 2/3 与 pred 的 2..5 **语义不对等**，请勿用「pred==tgt」对 tgt∈{2,3} interpret 为六类准确率。
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
        "note_eq_ep": "仅 tgt∈{0,1}；要求 pred_evt6 与 tgt 一致（与 ComCat 二分类严格口径一致）。",
        "contingency_true_pnw4_by_pred_evt6": {
            "true_pnw4_labels": list(PNW4),
            "pred_evt6_labels": list(EVT6),
            "counts_4x6": cm,
        },
        "row_normalized_4x6": row_norm,
        "support_per_true_pnw4": {PNW4[i]: row_sum[i] for i in range(4)},
        "note": "PNW4 与 EVT6 语义不同；2×6/3×6 列为「真值该类样本的预测分布」，非对角准确率。",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

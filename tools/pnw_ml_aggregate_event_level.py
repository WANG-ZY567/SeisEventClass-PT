#!/usr/bin/env python3
"""
波形级预测结果 -> 事件级聚合（多数票 / 概率平均，若提供 prob 列）。

输入 CSV 至少包含列：event_id, y_true, y_pred
可选列：y_prob_class1（二分类正类概率）或 prob_0..prob_K（多类）

输出：事件级 Accuracy / Macro-F1；可选 per-class F1（sklearn 可选）。
无 sklearn 时用宏平均近似：逐类 P/R/F1 由混淆矩阵手算。
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def macro_f1_from_confusion(cm: dict[tuple[int, int], int], n_class: int) -> float:
    f1s = []
    for c in range(n_class):
        tp = cm.get((c, c), 0)
        fp = sum(cm.get((cc, c), 0) for cc in range(n_class) if cc != c)
        fn = sum(cm.get((c, cc), 0) for cc in range(n_class) if cc != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def majority_vote(rows: list[dict]) -> tuple[int, int]:
    """返回 (y_true_event, y_pred_event)；假设同 event 的 y_true 一致。"""
    yt = int(rows[0]["y_true"])
    preds = [int(r["y_pred"]) for r in rows]
    cnt = Counter(preds)
    yp = cnt.most_common(1)[0][0]
    return yt, yp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", type=Path, required=True)
    ap.add_argument("--out_json", type=Path, default=None)
    ap.add_argument("--n_class", type=int, default=2)
    args = ap.parse_args()

    by_ev: dict[str, list[dict]] = defaultdict(list)
    with args.pred_csv.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            eid = (row.get("event_id") or "").strip()
            if not eid:
                continue
            by_ev[eid].append(row)

    cm = defaultdict(int)
    correct = 0
    for eid, rows in by_ev.items():
        yt, yp = majority_vote(rows)
        if yt == yp:
            correct += 1
        cm[(yt, yp)] += 1

    n_ev = len(by_ev)
    acc = correct / n_ev if n_ev else 0.0
    mf1 = macro_f1_from_confusion(cm, args.n_class)

    report = {
        "n_events": n_ev,
        "event_accuracy": acc,
        "event_macro_f1": mf1,
        "aggregation": "majority_vote_on_y_pred",
    }
    print(report)
    if args.out_json:
        import json

        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

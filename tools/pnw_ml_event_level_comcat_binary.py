#!/usr/bin/env python3
"""
从 test_results CSV（含 comcat_event_id, tgt_evt6, pred_evt6）生成事件级二分类指标。

规则：
  - 每个 event：真值 = mode(tgt_evt6)；预测 = mode(pred_evt6)
  - **子集口径**：仅 mode(pred)∈{0,1} 的事件计入 event_binary_accuracy_on_eq_ep_predictions
  - **全事件严格**：strict_event_binary_accuracy_all_events = 正确事件数 / n_events

输出 JSON（保留原扁平字段名以兼容旧文档）。
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def mode_val(xs: list[int]) -> int:
    return Counter(xs).most_common(1)[0][0]


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_csv", type=Path, required=True)
    ap.add_argument("--out_json", type=Path, required=True)
    ap.add_argument("--event_id_col", default="comcat_event_id")
    args = ap.parse_args()

    by_ev: dict[str, dict] = defaultdict(lambda: {"tgt": [], "pred": []})

    with args.results_csv.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            eid = (row.get(args.event_id_col) or "").strip()
            if not eid:
                continue
            by_ev[eid]["tgt"].append(int(float(row["tgt_evt6"])))
            by_ev[eid]["pred"].append(int(float(row["pred_evt6"])))

    evt_rows: list[tuple[str, int, int]] = []
    for eid, d in by_ev.items():
        yt = mode_val(d["tgt"])
        pr_mode = mode_val(d["pred"])
        evt_rows.append((eid, yt, pr_mode))

    pairs = [(t, p) for _, t, p in evt_rows if p in (0, 1)]
    correct = sum(1 for t, p in pairs if t == p)
    acc_subset = correct / len(pairs) if pairs else 0.0
    other_ev = sum(1 for *_, p in evt_rows if p not in (0, 1))

    strict_correct = sum(1 for _, t, p in evt_rows if p == t)
    n_ev = len(evt_rows)
    strict_acc = strict_correct / n_ev if n_ev else 0.0

    def f1_evt_full(pos: int) -> float:
        tp = sum(1 for _, t, p in evt_rows if t == pos and p == pos)
        fp = sum(1 for _, t, p in evt_rows if t != pos and p == pos)
        fn = sum(1 for _, t, p in evt_rows if t == pos and p != pos)
        return f1_from_counts(tp, fp, fn)

    f1_eq_s = f1_evt_full(0)
    f1_ep_s = f1_evt_full(1)
    macro_f1_s = (f1_eq_s + f1_ep_s) / 2.0

    report = {
        "n_events": n_ev,
        "event_binary_accuracy_on_eq_ep_predictions": acc_subset,
        "n_events_pred_resolved_eq_ep": len(pairs),
        "n_events_pred_other_class": other_ev,
        "note": "pred_evt6 mode not in {0,1} counted as other at event level（子集准确率分母不含 other）。",
        "strict_event_binary_accuracy_all_events": strict_acc,
        "strict_per_class_f1_eq0_expl1": {"eq_f1": f1_eq_s, "ep_f1": f1_ep_s},
        "strict_macro_f1_binary": macro_f1_s,
        "strict_note": "全事件分母 n_events；mode(pred) 必须等于 mode(tgt)。",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

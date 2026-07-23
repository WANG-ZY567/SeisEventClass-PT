#!/usr/bin/env python3
"""
ComCat external zero-shot: data `test_results_*_test.csv`(data pred_evt6, tgt_evt6), 
data"ComCat data"value: 

  - value: tgt_evt6 in {0,1}(earthquake=0, explosion=1), data prepare_comcat data meta. 

  - **data**: data pred_evt6in{0,1} data binary_accuracy / F1; pred_other data. 

  - **data**(data strict_*): data = data; data iff pred_evt6==tgt_evt6; 
    pred data 2..5 data; F1 data FN. 

value: JSON data --out_json(data/data). 
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_csv", type=Path, required=True)
    ap.add_argument("--out_json", type=Path, required=True)
    args = ap.parse_args()

    import csv

    y_true: list[int] = []
    y_pred_raw: list[int] = []
    y_pred_bin: list[int] = []
    pred_other = 0
    n = 0
    with args.results_csv.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            n += 1
            yt = int(float(row["tgt_evt6"]))
            pr = int(float(row["pred_evt6"]))
            y_true.append(yt)
            y_pred_raw.append(pr)
            if pr == 0:
                y_pred_bin.append(0)
            elif pr == 1:
                y_pred_bin.append(1)
            else:
                pred_other += 1
                y_pred_bin.append(-1)

    pairs = [(t, p) for t, p in zip(y_true, y_pred_bin) if p >= 0]

    correct = sum(1 for t, p in pairs if t == p)
    tot = len(pairs)
    acc = correct / tot if tot else 0.0

    def f1_bin(pos: int) -> float:
        tp = sum(1 for t, p in pairs if t == pos and p == pos)
        fp = sum(1 for t, p in pairs if t != pos and p == pos)
        fn = sum(1 for t, p in pairs if t == pos and p != pos)
        return f1_from_counts(tp, fp, fn)

    f1_eq = f1_bin(0)
    f1_ep = f1_bin(1)
    macro_f1 = (f1_eq + f1_ep) / 2.0

    strict_correct = sum(1 for t, pr in zip(y_true, y_pred_raw) if pr == t)
    strict_acc = strict_correct / n if n else 0.0

    def f1_bin_full_class(pos: int) -> float:
        tp = sum(1 for t, pr in zip(y_true, y_pred_raw) if t == pos and pr == pos)
        fp = sum(1 for t, pr in zip(y_true, y_pred_raw) if t != pos and pr == pos)
        fn = sum(1 for t, pr in zip(y_true, y_pred_raw) if t == pos and pr != pos)
        return f1_from_counts(tp, fp, fn)

    f1_eq_s = f1_bin_full_class(0)
    f1_ep_s = f1_bin_full_class(1)
    macro_f1_s = (f1_eq_s + f1_ep_s) / 2.0

    report = {
        "n_rows": n,
        "n_strict_pairs_pred_in_eq_ep": tot,
        "pred_evt6_not_eq_ep": pred_other,
        "frac_pred_other": pred_other / n if n else 0.0,
        "binary_accuracy_on_strict_subset": acc,
        "per_class_f1_eq0_expl1": {"eq_f1": f1_eq, "ep_f1": f1_ep},
        "macro_f1_binary": macro_f1,
        "note": "data pred_evt6in{0,1} data binary_accuracy / macro_f1; data pred_other. ",
        "strict_binary_accuracy_all_rows": strict_acc,
        "strict_per_class_f1_eq0_expl1": {"eq_f1": f1_eq_s, "ep_f1": f1_ep_s},
        "strict_macro_f1_binary": macro_f1_s,
        "strict_note": "data n_rows; pred_evt6 data tgt_evt6; pred data 2..5 data. ",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

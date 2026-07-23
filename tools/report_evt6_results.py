#!/usr/bin/env python3
"""
data test_results_*.csv data evt6 data(data): 
- Overall Accuracy(data)
- data Precision / Recall / F1
- Macro-F1
- data(csv + Markdown data)

data `training/validate.py` data `--save_test_results` value: 
  reports/<run>/test_results_diting2_evt6_test.csv
value: pred_evt6, tgt_evt6(data id)
"""

import argparse
import os
from datetime import datetime
from glob import glob
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd


# Open-source note: implementation detail.
ID2NAME: Dict[int, str] = {0: "eq", 1: "ep", 2: "co", 3: "sp", 4: "se", 5: "ot"}
NAME2ID: Dict[str, int] = {v: k for k, v in ID2NAME.items()}


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def _find_default_results_csv(run_dir: str) -> str:
    cands = sorted(
        glob(os.path.join(run_dir, "test_results_*_test.csv")) + glob(os.path.join(run_dir, "test_results_*.csv")),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    if not cands:
        raise FileNotFoundError(f"data run_dir data test_results_*.csv: {run_dir}")
    return cands[0]


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Tuple[float, np.ndarray, List[dict], float]:
    """
    Returns:
      acc, cm, per_class_rows, macro_f1
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    total = int(cm.sum())
    correct = int(np.trace(cm))
    acc = _safe_div(correct, total)

    per_class = []
    f1s = []
    for i in range(num_classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * prec * rec, prec + rec) if (prec + rec) > 0 else 0.0
        support = int(cm[i, :].sum())
        per_class.append(
            {
                "class_id": i,
                "class_name": ID2NAME.get(i, str(i)),
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "support": support,
            }
        )
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s)) if len(f1s) else 0.0
    return acc, cm, per_class, macro_f1


def _cm_to_markdown(cm: np.ndarray, names: List[str]) -> str:
    header = "| true\\pred | " + " | ".join(names) + " |\n"
    sep = "|---" + "|---" * (len(names) + 1) + "|\n"
    rows = []
    for i, n in enumerate(names):
        rows.append("| " + n + " | " + " | ".join(str(int(x)) for x in cm[i].tolist()) + " |")
    return header + sep + "\n".join(rows) + "\n"


def _cm_to_markdown_rect(cm: np.ndarray, true_names: List[str], pred_names: List[str]) -> str:
    """
    data confusion matrix: 
      - value: true_names(data cm.shape[0])
      - value: pred_names(data cm.shape[1])
    """
    if cm.shape[0] != len(true_names):
        raise ValueError(f"cm data true_names value: cm={cm.shape}, true_names={len(true_names)}")
    if cm.shape[1] != len(pred_names):
        raise ValueError(f"cm data pred_names value: cm={cm.shape}, pred_names={len(pred_names)}")
    header = "| true\\pred | " + " | ".join(pred_names) + " |\n"
    sep = "|---" + "|---" * (len(pred_names) + 1) + "|\n"
    rows = []
    for i, n in enumerate(true_names):
        rows.append("| " + n + " | " + " | ".join(str(int(x)) for x in cm[i].tolist()) + " |")
    return header + sep + "\n".join(rows) + "\n"


def _parse_class_list(s: str) -> List[int]:
    """
    value: 
      - data class name: eq,ep,co,sp,se,ot
      - data id: 0,1,2
    """
    s = (s or "").strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: List[int] = []
    for p in parts:
        if p.isdigit():
            out.append(int(p))
        else:
            if p not in NAME2ID:
                raise ValueError(f"value: {p}(value: {sorted(NAME2ID.keys())} data 0-5)")
            out.append(NAME2ID[p])
    # Open-source note: implementation detail.
    seen: Set[int] = set()
    uniq: List[int] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _compute_metrics_excluding_true(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, exclude_true_ids: List[int]
) -> Tuple[int, float, np.ndarray, List[dict], float, List[int]]:
    """
    data"data true label"value: 
      - data y_true data exclude_true_ids data
      - confusion matrix data pred data num_classes data(data"data se"data)
      - per-class / macro-f1 data true data(data 5-way macro-f1)

    Returns:
      kept_total, acc, cm_kept_rows (K x num_classes), per_class_rows(K), macro_f1, kept_true_ids
    """
    exclude_set = set(int(x) for x in exclude_true_ids)
    keep_mask = np.array([int(t) not in exclude_set for t in y_true.tolist()], dtype=bool)
    yt = y_true[keep_mask]
    yp = y_pred[keep_mask]
    kept_total = int(yt.shape[0])
    kept_true_ids = [i for i in range(num_classes) if i not in exclude_set]

    # Open-source note: implementation detail.
    acc = float(np.mean((yt == yp).astype(np.float64))) if kept_total > 0 else 0.0

    # full cm on filtered, then keep only selected true rows
    cm_full = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(yt.tolist(), yp.tolist()):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm_full[t, p] += 1
    cm_kept = cm_full[kept_true_ids, :]

    per_class: List[dict] = []
    f1s: List[float] = []
    for i in kept_true_ids:
        tp = int(cm_full[i, i])
        fp = int(cm_full[:, i].sum() - tp)  # Open-source note: implementation detail.
        fn = int(cm_full[i, :].sum() - tp)
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * prec * rec, prec + rec) if (prec + rec) > 0 else 0.0
        support = int(cm_full[i, :].sum())
        per_class.append(
            {
                "class_id": i,
                "class_name": ID2NAME.get(i, str(i)),
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "support": support,
            }
        )
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s)) if len(f1s) else 0.0
    return kept_total, acc, cm_kept, per_class, macro_f1, kept_true_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="", help="reports/<timestamp>_<model>_<dataset> data; data --results_csv")
    ap.add_argument("--results_csv", default="", help="test_results_*.csv data(data --run_dir data)")
    ap.add_argument("--out_md", default="", help="data Markdown data; data run_dir/EVT6_TEST_REPORT.md")
    ap.add_argument(
        "--exclude_true_classes",
        default="",
        help="data true label data(data, data eq,ep,co,sp,se,ot data id 0-5), value: se",
    )
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir) if args.run_dir else ""
    results_csv = os.path.abspath(args.results_csv) if args.results_csv else ""

    if results_csv:
        csv_path = results_csv
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)
        if not run_dir:
            run_dir = os.path.dirname(csv_path)
    else:
        if not run_dir:
            raise ValueError("data --results_csv data --run_dir")
        csv_path = _find_default_results_csv(run_dir)

    df = pd.read_csv(csv_path, low_memory=False)
    if "tgt_evt6" not in df.columns or "pred_evt6" not in df.columns:
        raise KeyError(f"CSV value: data tgt_evt6/pred_evt6, data={list(df.columns)}")

    y_true = df["tgt_evt6"].astype(int).to_numpy()
    y_pred = df["pred_evt6"].astype(int).to_numpy()

    num_classes = 6
    acc, cm, per_class, macro_f1 = _compute_metrics(y_true, y_pred, num_classes=num_classes)
    names = [ID2NAME[i] for i in range(num_classes)]
    exclude_true_ids = _parse_class_list(args.exclude_true_classes)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_md = os.path.abspath(args.out_md) if args.out_md else os.path.join(run_dir, "EVT6_TEST_REPORT.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)

    # Open-source note: implementation detail.
    cm_csv = os.path.join(os.path.dirname(out_md), "confusion_matrix_evt6.csv")
    pd.DataFrame(cm, index=[f"true_{n}" for n in names], columns=[f"pred_{n}" for n in names]).to_csv(cm_csv)

    lines = []
    lines.append("# EVT6 Test Report\n\n")
    lines.append(f"- value: {now}\n")
    lines.append(f"- value: `{csv_path}`\n")
    lines.append(f"- value: `{out_md}`\n")
    lines.append(f"- metadata CSV: `{cm_csv}`\n")
    lines.append("\n---\n\n")

    lines.append("## Overall metrics\n\n")
    lines.append(f"- **Accuracy**: {acc:.4f}\n")
    lines.append(f"- **Macro-F1**: {macro_f1:.4f}\n")
    lines.append("\n---\n\n")

    lines.append("## Per-class metrics\n\n")
    lines.append("| class | support | precision | recall | f1 |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for r in per_class:
        lines.append(
            f"| {r['class_name']} | {int(r['support'])} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |\n"
        )
    lines.append("\n---\n\n")

    lines.append("## 3. Confusion Matrix(true x pred)\n\n")
    lines.append(_cm_to_markdown(cm, names))

    if exclude_true_ids:
        kept_total, acc_ex, cm_kept, per_class_ex, macro_f1_ex, kept_true_ids = _compute_metrics_excluding_true(
            y_true, y_pred, num_classes=num_classes, exclude_true_ids=exclude_true_ids
        )
        kept_true_names = [ID2NAME[i] for i in kept_true_ids]
        exclude_true_names = [ID2NAME.get(i, str(i)) for i in exclude_true_ids]

        cm_csv_ex = os.path.join(os.path.dirname(out_md), f"confusion_matrix_evt6_excl-{'-'.join(exclude_true_names)}.csv")
        pd.DataFrame(
            cm_kept, index=[f"true_{n}" for n in kept_true_names], columns=[f"pred_{n}" for n in names]
        ).to_csv(cm_csv_ex)

        lines.append("\n---\n\n")
        lines.append("## Event-level metrics\n\n")
        lines.append(f"- data true value: `{', '.join(exclude_true_names)}`\n")
        lines.append(f"- value: {kept_total}\n")
        lines.append(f"- **Accuracy(exclude true={','.join(exclude_true_names)})**: {acc_ex:.4f}\n")
        lines.append(f"- **Macro-F1(exclude true={','.join(exclude_true_names)})**: {macro_f1_ex:.4f}\n")
        lines.append(f"- metadata CSV(data true data, pred data 6 data): `{cm_csv_ex}`\n")
        lines.append("\n\n")

        lines.append("## Confusion matrix\n\n")
        lines.append("| class | support | precision | recall | f1 |\n")
        lines.append("|---|---:|---:|---:|---:|\n")
        for r in per_class_ex:
            lines.append(
                f"| {r['class_name']} | {int(r['support'])} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |\n"
            )
        lines.append("\n\n")

        lines.append("## Error examples\n\n")
        lines.append(_cm_to_markdown_rect(cm_kept, kept_true_names, names))

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("".join(lines))

    print(f"[OK] report: {out_md}")
    print(f"[OK] confusion matrix csv: {cm_csv}")


if __name__ == "__main__":
    main()




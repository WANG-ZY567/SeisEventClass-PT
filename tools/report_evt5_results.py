#!/usr/bin/env python3
"""
从 test_results_*.csv 生成 evt5（剔除 se）分类结果报告：
- Overall Accuracy
- 每类别 Precision / Recall / F1
- Macro-F1
- 混淆矩阵（csv + Markdown 表格）

要求 CSV 至少包含列：pred_evt5, tgt_evt5（整数类别 id）
"""

import argparse
import os
from datetime import datetime
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# evt5 定义：0 eq,1 ep,2 co,3 sp,4 ot
ID2NAME: Dict[int, str] = {0: "eq", 1: "ep", 2: "co", 3: "sp", 4: "ot"}


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def _find_default_results_csv(run_dir: str) -> str:
    cands = sorted(
        glob(os.path.join(run_dir, "test_results_*_test.csv")) + glob(os.path.join(run_dir, "test_results_*.csv")),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    if not cands:
        raise FileNotFoundError(f"在 run_dir 未找到 test_results_*.csv：{run_dir}")
    return cands[0]


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Tuple[float, np.ndarray, List[dict], float]:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="", help="reports/<timestamp>_<model>_<dataset> 目录；留空则必须提供 --results_csv")
    ap.add_argument("--results_csv", default="", help="test_results_*.csv 路径（优先于 --run_dir 自动探测）")
    ap.add_argument("--out_md", default="", help="输出 Markdown 路径；默认写入 run_dir/EVT5_TEST_REPORT.md")
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
            raise ValueError("请提供 --results_csv 或 --run_dir")
        csv_path = _find_default_results_csv(run_dir)

    df = pd.read_csv(csv_path, low_memory=False)
    if "tgt_evt5" not in df.columns or "pred_evt5" not in df.columns:
        raise KeyError(f"CSV 缺少列：需要 tgt_evt5/pred_evt5，实际列={list(df.columns)}")

    y_true = df["tgt_evt5"].astype(int).to_numpy()
    y_pred = df["pred_evt5"].astype(int).to_numpy()

    num_classes = 5
    acc, cm, per_class, macro_f1 = _compute_metrics(y_true, y_pred, num_classes=num_classes)
    names = [ID2NAME[i] for i in range(num_classes)]

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_md = os.path.abspath(args.out_md) if args.out_md else os.path.join(run_dir, "EVT5_TEST_REPORT.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)

    cm_csv = os.path.join(os.path.dirname(out_md), "confusion_matrix_evt5.csv")
    pd.DataFrame(cm, index=[f"true_{n}" for n in names], columns=[f"pred_{n}" for n in names]).to_csv(cm_csv)

    lines = []
    lines.append("# EVT5 分类测试报告（exclude se；Accuracy / Macro-F1 / Confusion Matrix）\n\n")
    lines.append(f"- 生成时间：{now}\n")
    lines.append(f"- 输入结果：`{csv_path}`\n")
    lines.append(f"- 输出报告：`{out_md}`\n")
    lines.append(f"- 混淆矩阵 CSV：`{cm_csv}`\n")
    lines.append("\n---\n\n")

    lines.append("## 1. Overall 指标\n\n")
    lines.append(f"- **Accuracy**：{acc:.4f}\n")
    lines.append(f"- **Macro-F1**：{macro_f1:.4f}\n")
    lines.append("\n---\n\n")

    lines.append("## 2. 各类别指标\n\n")
    lines.append("| class | support | precision | recall | f1 |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for r in per_class:
        lines.append(f"| {r['class_name']} | {int(r['support'])} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |\n")
    lines.append("\n---\n\n")

    lines.append("## 3. Confusion Matrix（true × pred）\n\n")
    lines.append(_cm_to_markdown(cm, names))

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("".join(lines))

    print(f"[OK] report: {out_md}")
    print(f"[OK] confusion matrix csv: {cm_csv}")


if __name__ == "__main__":
    main()


